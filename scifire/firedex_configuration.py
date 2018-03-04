from scale_client.stats.random_variable import RandomVariable
from scifire.firedex_scenario import FiredexScenario
import logging
log = logging.getLogger(__name__)


class FiredexConfiguration(FiredexScenario):
    """
    Represents a specific configuration for a scenario i.e. subscriber/publishers' actual topics, topics' characteristics,
    network channel state, etc. at a particular point in time.
    """

    def __init__(self, draw_subscriptions_from_advertisements=True, **kwargs):
        super(FiredexConfiguration, self).__init__(**kwargs)

        # avoid subscribing to non-advertised topics
        self.draw_subscriptions_from_advertisements = draw_subscriptions_from_advertisements

        # these will get filled in later
        self.pub_rates = []
        self._data_sizes = dict()
        self.service_rates = []
        self._advertisements = None
        self.subscriptions = None
        self._network_flows = None

    def generate_configuration(self):
        """
        Using the specified random variable distributions and static configuration parameters, generates an actual
        configuration for one simulation run of an experiment that includes assigned values (be they static or
        further random distributions) pulled from probability distributions for specific topic publication rates,
        event sizes, utility functions, subscriptions, etc.  This configuration can be used to run a simulation, drive
        the algorithms, or configure clients in an emulated experiment.
        """

        data_size_rvs = [RandomVariable.build(kws) for kws in self.topic_class_data_sizes]
        pub_rate_rvs = [RandomVariable.build(kws) for kws in self.topic_class_pub_rates]
        # TODO: make these dicts instead of lists?
        self.service_rates = []
        self.pub_rates = []
        for topics, size_rv, rate_rv in zip(self.topic_classes, data_size_rvs, pub_rate_rvs):
            for t in topics:
                self._data_sizes[t] = data_size = size_rv.get_int()
                # TODO: handle specifying different bandwidth values
                srv_rate = self.calculate_service_rate(data_size)
                self.service_rates.append(srv_rate)

                pub_rate = rate_rv.get()
                self.pub_rates.append(pub_rate)

        assert len(self.service_rates) == self.num_topics
        assert len(self.pub_rates) == self.num_topics

        # Need to generate advertisements first in case we base subscriptions on them
        self.generate_advertisements()
        self.generate_subscriptions()

    def generate_subscriptions(self):
        """
        Generates the topic subscriptions for this scenario configuration.

        NOTE: this is currently only for a single subscriber and the subscriptions are assumed to remain constant
        for the entire experiment's duration.

        :return: list of topics subscribed to
        """

        # TODO: generate subs for multiple subscribers
        # TODO: need to distinguish IC from other FFs when we do this

        # IDEA: for a single subscriber, consider each topic class and generate topics from the given distribution
        # until we have a number of unique subscriptions >= #requested for that class
        # TODO: need to bring in topic start time and duration distributions too... maybe make subscriptions a class???
        # ENHANCE: maybe we should actually have the # subscriptions be a RV so we can vary the subscribers that way?

        subs = []
        for class_idx, (class_topic_rate, rv) in enumerate(zip(self.topic_class_sub_rates, self.topic_class_sub_dists)):
            # to reduce variance in the simulator due to subscriptions for non-advertised topics, we draw
            # subscriptions only from those that are advertised unless otherwise explicitly requested.
            if self.draw_subscriptions_from_advertisements:
                topic_choices = [ad for ad in self.advertised_topics if self.class_for_topic(ad) == class_idx]
            else:
                topic_choices = self.topics_for_class(class_idx)

            ntopics = int(class_topic_rate * len(topic_choices))
            # XXX: even if config looks okay, IC might request too many topics for a class
            ntopics = min(len(topic_choices), ntopics)

            rv = self.build_sampling_random_variable(rv, len(topic_choices))
            try:
                class_subs = rv.sample(topic_choices, ntopics)
                subs.extend(class_subs)
            except ValueError as e:
                log.error("failed to generate topics for class %d due to error: %s" % (class_idx, e))

        self.subscriptions = subs
        if not self.subscriptions:
            log.error("failed to generate any subscriptions!  Check your topic_class_sub_<rates|dists> !!")
        else:
            log.debug("topic subscriptions: %s" % self.subscriptions)
        return subs

    def generate_advertisements(self):
        """
        Generates the topics each publisher will publish to for this scenario configuration.
        :return: self.advertisements
        """

        # ENHANCE: try to skew ads so that FFs tend to publish to the same topics that IoT devs do not e.g. FF health monitoring
        # ENHANCE: use lists nested inside dicts instead of two lists of lists?

        ff_ads = dict()
        iot_ads = dict()

        ads_rvs = [self.build_sampling_random_variable(dist, len(self.topics_for_class(tc_idx))) for tc_idx, dist in enumerate(self.topic_class_pub_dists)]

        # For each type of publisher e.g. FF or IoT-dev
        for (num_pubs, tc_num_ads, tc_ads_rvs, ads) in ((self.num_ffs, self.topic_class_advertisements_per_ff, ads_rvs, ff_ads),
                                                        (self.num_iots, self.topic_class_advertisements_per_iot, ads_rvs, iot_ads)):
            # generate ads for each publisher
            for p in range(num_pubs):
                ads_for_pub = []
                # based on each topic class distribution/population
                for tc in range(self.ntopic_classes):
                    rv = ads_rvs[tc]
                    # TODO: pull this num from a RV dist rather than assuming it's a constant
                    num_ads = tc_num_ads[tc]
                    possible_ads = list(self.topics_for_class(tc))
                    # XXX: ensure we don't request too many samples from population
                    num_ads = min(num_ads, len(possible_ads))
                    try:
                        ads_for_pub.extend(rv.sample(possible_ads, num_ads))
                    except ValueError as e:
                        log.error("failed to generate advertisements for class %d due to error: %s" % (tc, e))
                ads[p] = ads_for_pub

        log.debug("FF advertisements: %s" % ff_ads)
        log.debug("IoT advertisements: %s" % iot_ads)

        self._advertisements = (ff_ads, iot_ads)
        return self.advertisements

    @property
    def advertisements(self):
        """
        :return: (ff_ads, iot_ads) where e.g. ff_ads is a dict of lists mapping FF publisher to its topics advertised (published)
        :rtype: tuple[dict[list[str|int]]]
        """
        return self._advertisements

    @property
    def advertised_topics(self):
        """
        :returns: a set of all topics advertised by ANY publisher
        """
        return set(ad for pub_class in self.advertisements for ads in pub_class.itervalues() for ad in ads)

    def calculate_service_rate(self, pkt_size, bandwidth=None):
        """
        Calculates the transmission rate of a packet on the network according to the specified bandwidth.
        :param pkt_size: in bytes
        :param bandwidth: defaults to self.bandwidth
        :return:
        """
        # TODO: consider packet (header) overhead when doing this!
        if bandwidth is None:
            bandwidth = self.bandwidth
        return self.bandwidth_bytes(bandwidth) / float(pkt_size)

    def build_sampling_random_variable(self, rv_dist_cfg, population_size):
        """
        Returns a RandomVariable that is more likely to guarantee it will be able to properly sample from a population of
        the given size.

        It checks if the distribution has an upper bound; if it's the uniform distribution with no upper bound arg
        specified it will set them to the default of [low, population_size] where low is the lower bound specified else 0.
        This allows us to just specify a distribution as uniform when configuring the experiment and let the range be
        dynamically generated according to e.g. num_topics

        Also checks for proper lower bound e.g. zipf gets an additional argument to shift the range down to range [0, inf)

        :param rv_dist_cfg:
        :param population_size:
        :return:
        """

        # ENHANCE: store the random variable so we can keep it between runs?  would need to identify it e.g. with a string...

        rv = RandomVariable.build(rv_dist_cfg)
        if rv.is_upper_bounded():
            if rv.dist == 'uniform':
                args = list(rv_dist_cfg.get('args', [0]))
                if len(args) == 1:
                    low = args[0]
                    args.append(population_size - low)
                    rv_dist_cfg['args'] = args
                    rv = RandomVariable.build(rv_dist_cfg)

        if rv.dist == 'zipf':
            args = rv_dist_cfg.get('args')
            # XXX: shift if with the loc parameter
            if 'loc' not in rv_dist_cfg and (not args or len(args) <= 1):
                args = rv_dist_cfg.copy()
                args['loc'] = -1
                rv = RandomVariable.build(args)

        return rv

    @property
    def prio_classes(self):
        """
        Returns a list of the distinct priority classes where the first item is the highest priority.
        :return:
        :rtype: list[int]
        """
        return range(self.num_priority_levels)

    @property
    def net_flows(self):
        """
        Returns a list of the network flows.  Don't make assumption about what each network flow object is!  It may
        just be an int (simulation) or it may be a more complex object with information about e.g. port numbers.
        :return:
        """
        # let's at least have a default for the most basic simulated experiments
        if self._network_flows is None:
            return range(self.num_net_flows)
        return list(self._network_flows)

    @property
    def data_sizes(self):
        return self._data_sizes.values()

class QueueStabilityError(ValueError):
    """
    Generated when the check for ro values fails: this experimental run should be recorded as a wash.
    """
    pass
