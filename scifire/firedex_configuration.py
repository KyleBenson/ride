from scale_client.stats.random_variable import RandomVariable
from scifire.firedex_scenario import FiredexScenario

from collections import namedtuple
import itertools
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
        # these are indexed by hosts i.e. publishers or subscribers
        self._advertisements = None  # type: tuple[dict]
        self._subscriptions = None  # type: list[FiredexConfiguration.Subscription]
        # TODO: should probably do a similar simple object for network flows to keep matched with subscriber
        self._network_flows = None  # type: dict
        # also keep a backwards lookup since each network is unique to a subscriber
        self._net_flow_subscriber_map = None  # type: dict

    def generate_configuration(self):
        """
        Using the specified random variable distributions and static configuration parameters, generates an actual
        configuration for one simulation run of an experiment that includes assigned values (be they static or
        further random distributions) pulled from probability distributions for specific topic publication rates,
        event sizes, utility functions, subscriptions, etc.  This configuration can be used to run a simulation, drive
        the algorithms, or configure clients in an emulated experiment.
        """

        data_size_rvs = [self.build_random_variable(kws) for kws in self.topic_class_data_sizes]
        pub_rate_rvs = [self.build_random_variable(kws) for kws in self.topic_class_pub_rates]
        # TODO: make these dicts instead of lists?
        self.service_rates = []
        self.pub_rates = []
        for topics, size_rv, rate_rv in zip(self.topics_per_class, data_size_rvs, pub_rate_rvs):
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
        # This generates utilities too.
        self.generate_subscriptions()

    ## use this as a first version of a potential future object with more attributes e.g. start/end_time etc.
    Subscription = namedtuple('Subscription', ['subscriber', 'topic', 'utility_weight'])

    def generate_subscriptions(self):
        """
        Generates the subscriptions for this scenario configuration: each subscription matches a subscriber to one of
        the topics topics it's interested in and a utility function to capture the level of interest / value of info.

        NOTE: the subscriptions are assumed to remain constant for the entire experiment's duration.
        NOTE: the utility function is just a weight (i.e. all the same assumed log(1+lambda)-based function)

        :return: self.subscriptions
        """

        # IDEA: for a single subscriber, consider each topic class and generate topics from the given distribution
        # until we have a number of unique subscriptions >= #requested for that class
        # TODO: need to bring in topic start time and duration distributions too... maybe make subscriptions a class???
        # ENHANCE: maybe we should actually have the # subscriptions be a RV so we can vary the subscribers that way?

        self._subscriptions = []
        for subscriber in self.subscribers:
            subs = []
            for class_idx, (class_topic_rate, rv_top, rv_uw) in enumerate(zip(self.topic_class_sub_rates, self.topic_class_sub_dists,
                                                                   self.topic_class_utility_weights)):
                # to reduce variance in the simulator due to subscriptions for non-advertised topics, we draw
                # subscriptions only from those that are advertised unless otherwise explicitly requested.
                if self.draw_subscriptions_from_advertisements:
                    topic_choices = [ad for ad in self.advertised_topics if self.class_for_topic(ad) == class_idx]
                else:
                    topic_choices = self.topics_for_class(class_idx)

                ntopics = int(class_topic_rate * len(topic_choices))
                # distinguish IC from other FFs
                if subscriber == self.icp:
                    ntopics *= self.ic_sub_rate_factor
                # XXX: even if config looks okay, IC might request too many topics for a class
                ntopics = min(len(topic_choices), ntopics)

                rv_top = self.build_sampling_random_variable(rv_top, len(topic_choices))
                rv_uw = self.build_sampling_random_variable(rv_uw, len(topic_choices))

                try:
                    class_subs = rv_top.sample(topic_choices, ntopics)
                    class_utils = [rv_uw.get() for s in class_subs]

                    class_subs = [FiredexConfiguration.Subscription(subscriber, s, utility_weight=uw) for s, uw in zip(class_subs, class_utils)]
                    subs.extend(class_subs)
                except ValueError as e:
                    log.error("failed to generate topics for class %d due to error: %s" % (class_idx, e))

            self._subscriptions.extend(subs)

        if not self._subscriptions:
            log.error("failed to generate any subscriptions!  Check your topic_class_sub_<rates|dists> !!")
        else:
            log.debug("topic subscriptions: %s" % self.subscriptions)
        return self.subscriptions

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

    ####   HELPER FUNCTIONS:

    # NOTE: originally wrote everything to consider topics, but extending it to consider multiple subscribers proved
    #    we needed to just consider subscriptions directly since otherwise we need to pass around the subscriber
    #    everywhere and it could get separated from its corresponding topic(s) easily (esp. since we just use lists).

    @property
    def subscription_topics(self):
        """
        :return: the list of ALL subscriptions for ALL subscribers.
        :rtype: list[str|int]
        """
        return [sub.topic for sub in self.subscriptions]

    @property
    def subscriptions(self):
        """
        Return all Subscription objects active in the system.
        :rtype: list[FiredexConfiguration.Subscription]
        """
        return self._subscriptions

    def get_subscription_topics(self, subscriber=None):
        """
        Return the list of subscription topics for all subscribers or just the given subscriber if specified.
        :param subscriber:
        :return:
        :rtype: list[str|int]
        """
        return [sub.topic for sub in self.get_subscriptions(subscriber)]

    def get_subscriptions(self, subscriber=None):
        """
        Return the list of subscriptions for all subscribers or just the given subscriber if specified.
        :param subscriber:
        :return:
        :rtype: list[FiredexConfiguration.Subscription]
        """

        if subscriber is None:
            return self.subscriptions
        # ENHANCE: use a dict for quicker lookup?  probably not bother as that'd make dynamics a huge pain...
        return [sub for sub in self.subscriptions if sub.subscriber == subscriber]

    # DEPRECATED: Mostly kept around for a bunch of older test cases
    def get_utility_weight(self, topic, subscriber=None):
        """
        Returns the utility weight of the specified topic and subscriber.  If the subscriber has not subscribed to this
        topic, returns 0.

        :param topic:
        :param subscriber: which subscriber to find the utilities of (default=pick any arbitrarily)
        :return:
        """

        if subscriber is None:
            subscriber = self.arbitrary_subscriber

        for sub in self.get_subscriptions(subscriber):
            if sub.topic == topic:
                return sub.utility_weight
        else:  # must not be subscribed to
            return 0

    @property
    def subscription_utility_weights(self):
        return [sub.utility_weight for sub in self.subscriptions]

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

    def topics_to_subscriptions(self, topic_values):
        """
        Converts a per-topic vector (list) of values into a corresponding per-subscription vector (list) of the
        corresponding values.
        :param topic_values:
        :return:
        """

        # ASSUMPTION: we can index a topic vector with the topic values contained in the list self.subscriptions
        #   This is trivial as long as we keep lists of enumerated topics (i.e. range(ntopics));
        #   otherwise we might need to use dicts if we move to strings for topics.

        return [topic_values[top] for top in self.subscription_topics]

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
        :rtype: RandomVariable
        """

        # ENHANCE: store the random variable so we can keep it between runs?  would need to identify it e.g. with a string...

        # make a copy so we don't change the configuration dict!
        if isinstance(rv_dist_cfg, dict):
            rv_dist_cfg = rv_dist_cfg.copy()
        # XXX: ensure we have a dict so we can change the params without causing TypeErrors!
        else:
            rv_dist_cfg = RandomVariable.expand_config(rv_dist_cfg)

        rv = RandomVariable.build(rv_dist_cfg)
        if rv.is_upper_bounded():
            if rv.dist == 'uniform':
                args = list(rv_dist_cfg.get('args', [0]))
                if len(args) == 1:
                    low = args[0]
                    args.append(population_size - low)
                    rv_dist_cfg['args'] = args
                    rv = self.build_random_variable(rv_dist_cfg)

        if rv.dist == 'zipf':
            args = rv_dist_cfg.get('args')
            # XXX: shift if with the loc parameter
            if 'loc' not in rv_dist_cfg and (not args or len(args) <= 1):
                args = rv_dist_cfg.copy()
                args['loc'] = -1
                rv = self.build_random_variable(args)

        return rv

    def build_random_variable(self, rv_dist_cfg):
        """
        Just builds a random variable currently, but may store it for re-use in the future so use this to build them!
        :param rv_dist_cfg: RandomVariable configuration
        :return:
        :rtype: RandomVariable
        """
        return RandomVariable.build(rv_dist_cfg)

    @property
    def prio_classes(self):
        """
        Returns a list of the distinct priority classes where the first item is the highest priority.
        Note that if self.num_priority_levels == 0, the returned list contains only the null value cls.NO_PRIORITY
        :return:
        :rtype: list[int]
        """
        if self.num_priority_levels > 0:
            return range(self.num_priority_levels)
        else:
            return tuple((FiredexConfiguration.NO_PRIORITY,))

    NO_PRIORITY = -1

    @property
    def net_flows(self):
        """
        Returns a list of all network flows for all subscribers.
        Don't make assumption about what each network flow object is!  It may
        just be an int (simulation) or it may be a more complex object with information about e.g. port numbers.
        :return:
        :rtype: list
        """
        return list(itertools.chain(*[self.net_flows_for_subscriber(sub) for sub in self.subscribers]))

    def net_flows_for_subscriber(self, subscriber):
        """
        Returns a list of network flows associated with the requested subscriber.
        :param subscriber:
        :return:
        """
        self.__ensure_net_flows_generated()
        return self._network_flows[subscriber]

    def subscriber_for_flow(self, net_flow):
        """
        :param net_flow:
        :returns: the subscriber associated with the given network flow
        """
        self.__ensure_net_flows_generated()
        return self._net_flow_subscriber_map[net_flow]

    def __ensure_net_flows_generated(self, force_regeneration=False):
        """
        For simulated versions that don't specify the network flows but rather generate them just based on the #flows
        requested per subscriber, this method handles the actual generation to ensure other methods using the underlying
        map(s) will find actual network flows as expected.
        :param force_regeneration: if explicitly set to True, will re-generate flows even if already done before
        :return:
        """

        # for basic simulated version, ensure we have some defaults by just filling in the dict if we haven't yet
        if force_regeneration or self._network_flows is None:
            self._network_flows = dict()
            self._net_flow_subscriber_map = dict()
            for i, sub in enumerate(self.subscribers):
                flows = range(i*self.num_net_flows, (i+1)*self.num_net_flows)
                self._network_flows[sub] = flows

                # TODO: make a setter function to handle this correctly so if we receive net flows from the sub we bookkeep correctly?
                # set up a backwards lookup
                for f in flows:
                    self._net_flow_subscriber_map[f] = sub

    @property
    def data_sizes(self):
        return self._data_sizes.values()


class QueueStabilityError(ValueError):
    """
    Generated when the check for ro values fails: this experimental run should be recorded as a wash.
    """
    pass
