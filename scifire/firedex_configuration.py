from scale_client.stats.random_variable import RandomVariable
from scifire.firedex_scenario import FiredexScenario
import logging
log = logging.getLogger(__name__)


class FiredexConfiguration(FiredexScenario):
    """
    Represents a specific configuration for a scenario i.e. subscriber/publishers' actual topics, topics' characteristics,
    network channel state, etc. at a particular point in time.
    """

    def __init__(self, **kwargs):
        super(FiredexConfiguration, self).__init__(**kwargs)

        # these will get filled in later
        self.pub_rates = []
        self.data_sizes = dict()
        self.service_rates = []
        self.advertisements = None
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
                self.data_sizes[t] = data_size = size_rv.get_int()
                # TODO: handle specifying different bandwidth values
                srv_rate = self.calculate_service_rate(data_size)
                self.service_rates.append(srv_rate)

                pub_rate = rate_rv.get()
                self.pub_rates.append(pub_rate)

        assert len(self.service_rates) == self.num_topics
        assert len(self.pub_rates) == self.num_topics

        self.generate_subscriptions()
        self.generate_advertisements()

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
            rv = self.build_sampling_random_variable(rv, len(self.topics_for_class(class_idx)))
            ntopics = int(class_topic_rate * self.ntopics_per_class[class_idx])
            try:
                class_subs = rv.sample(self.topics_for_class(class_idx), ntopics)
                subs.extend(class_subs)
            except ValueError as e:
                log.error("failed to generate topics for class %d due to error: %s" % (class_idx, e))

        self.subscriptions = subs
        return subs

    def generate_advertisements(self):
        """
        Generates the topics each publisher will publish to for this scenario configuration.
        :return: (ff_ads, iot_ads) where ads is a list of lists mapping publisher to its topics advertised
        """

        # ENHANCE: try to skew ads so that FFs tend to publish to the same topics that IoT devs do not e.g. FF health monitoring
        # ENHANCE: use lists nested inside dicts instead of two lists of lists?

        ff_ads = []
        iot_ads = []

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
                    try:
                        ads_for_pub.extend(rv.sample(self.topics_for_class(tc), num_ads))
                    except ValueError as e:
                        log.error("failed to generate advertisements for class %d due to error: %s" % (tc, e))
                ads.append(ads_for_pub)

        log.debug("FF advertisements: %s" % ff_ads)
        log.debug("IoT advertisements: %s" % iot_ads)

        self.advertisements = (ff_ads, iot_ads)
        return ff_ads, iot_ads

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
