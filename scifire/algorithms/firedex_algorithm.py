from ..firedex_configuration import FiredexConfiguration, QueueStabilityError
from collections import namedtuple
from ..utilities import calculate_utility
from ..defaults import *

import logging
log = logging.getLogger(__name__)


class FiredexAlgorithm(object):
    """
    Abstract base class for assigning topic priorities (i.e. topic-> net flow -> prio mappings) according to the current
    (estimated or theoretical) system configuration/state. Derived classes will share the same analytical model that
    uses our queuing network model, but will calculate the mappings differently.

    Note that these mappings are stored in this class as state and will only be updated for a given configuration if
    explicitly requested!
    """

    def __init__(self, drop_policy='expon', ro_tolerance=DEFAULT_RO_TOLERANCE, **kwargs):
        """
        :param drop_policy: the default preemptive drop rate policy to apply after running the algorithm, which may be
            ignored by algorithms that implement more sophisticated policies
        :param ro_tolerance: configures how much ro values should be <1 e.g. verifies sum(ros) < 1.0 - ro_tolerance
            A positive value ensures even greater queue stability; a negative value violates this condition and
            allow queues to grow without bound (although this will result in completely wrong analytical results) (default=0)
        :param kwargs: passed to super constructor for possible multiple inheritance
        """

        # XXX: multiple inheritance
        try:
            super(FiredexAlgorithm, self).__init__(**kwargs)
        except TypeError:
            super(FiredexAlgorithm, self).__init__()

        # NOTE: we maintain a set of mappings and other state for each configuration being managed (i.e. broker and its
        # network ) AND for each subscriber connected with that broker.

        # these are filled in by _run_algorithm()
        self._topic_flow_map = dict()
        self._flow_prio_map = dict()

        self.ro_tolerance = ro_tolerance
        self.drop_policy = drop_policy
        # maps flows to drop rates in range [0,1] for a particular configuration; filled in when algorithm runs
        self._drop_rates = dict()

    ### Analytical model for queueing network
    ## NOTE: this model considers 4 queues:
    #  1) broker input queue for sorting/routing topics
    #  2) broker output queue for transmitting packets via network on different network flows
    #  3) SDN switch input queue for prioritization and dropping/bandwidth assignment by network flow
    #  4) SDN switch output queue (multi-class) for determining transmission rates of different topics according to the bandwidth

    ## Define the model as namedtuples to help ease working with it and extending it in the future
    # broker_out == switch_in generally, but we might add e.g. network drop rates.
    # broker/switch_thru refers to the arrival rate at the second queue of the corresponding layer
    Lambdas = namedtuple('Lambdas', ['broker_in', 'broker_thru', 'broker_out', 'switch_in', 'switch_thru', 'switch_out', 'sub_in'])
    # No 'thru's in the Mus since it's only defined on each queue's "server"
    Mus = namedtuple('Mus', ['broker_in', 'broker_out', 'switch_in', 'switch_out'])
    Ros = namedtuple('Ros', ['broker_in', 'broker_out', 'switch_in', 'switch_out'])

    def service_rates(self, configuration, subscriber=None):
        """
        Returns the expected service rates (MUs) at each queue for all topics given the configuration and
        optionally-specified subscriber.
        :param configuration:
        :param subscriber:
        :return:
        :rtype: FiredexAlgorithm.Mus
        """

        mus_switch_out = [configuration.calculate_service_rate(pkt_size) for pkt_size in configuration.data_sizes]

        return FiredexAlgorithm.Mus([DEFAULT_MU] * configuration.num_topics, [DEFAULT_MU] * configuration.num_topics,
                                    [DEFAULT_MU] * configuration.num_topics, mus_switch_out)

    def total_delays(self, configuration, subscriber=None):
        """
        Calculates the end-to-end delay of each topic on its route from the publisher(s) to the optionally-specified
        subscriber.  This includes queuing/service delays as well as network propagation delay (latency).

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: the subscriber we calculate delays for
        :return: list of lists of service delays where each outer index corresponds to the
                topic sharing that index in config.topics
        :rtype: list[float]
        """

        # TODO: consider pub-to-broker (averaged over pubs?) and broker-to-switch latency too?  maybe re-tx delay?
        return [configuration.latency + d for d in self.service_delays(configuration, subscriber=subscriber)]

    # ENHANCE: consider publisher-to-broker delay

    def service_delays(self, configuration, subscriber=None):
        """
        Calculates the service delay of each topic for each queue we model in our system.  Only considers the queues
        on route to the optionally-specified subscriber.

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: the subscriber we calculate delays for
        :return: list of service delays where each index corresponds to the topic sharing that index in config.topics
        :rtype: list[float]
        """

        lambdas = self.arrival_rates(configuration, subscriber=subscriber)
        mus = self.service_rates(configuration, subscriber=subscriber)

        # First the broker delays:
        # Each are MM1 queues
        lam_b_in = lambdas.broker_in
        mu_b_in = mus.broker_in[0]  # independent of topic!
        delta_b_in = [((1.0/mu_b_in)/(1-lam/mu_b_in)) for lam in lam_b_in]

        lam_b_thru = lambdas.broker_thru
        mu_b_out = mus.broker_out[0]  # independent of topic!
        delta_b_out = [((1.0/mu_b_out)/(1-lam/mu_b_out)) for lam in lam_b_thru]

        # Then the SDN switch delays:
        lam_s_in = lambdas.switch_in
        mu_s_in = mus.switch_in[0]  # independent of topic!

        # First, the priority queue needs to consider each priority class according to the network flow/priority mappings.
        # So we get the total rates for each of these classes first.
        topic_prios = self.get_topic_priorities(configuration, subscriber=subscriber)
        lam_topics = zip(lam_s_in, configuration.topics)
        lam_prios = [sum(lam if topic_prios[top] == p else 0.0 for lam, top in lam_topics) for p in configuration.prio_classes]
        num = sum(lam_prios)

        delta_s_in = [(num / ((mu_s_in - sum(lam_prios[:topic_prios[top]])) *
                              (mu_s_in - sum(lam_prios[: max(topic_prios[top] - 1, 0)])))  # prio=0 --> -1 index --> bad!
                       + 1.0/mu_s_in) for lam, top in lam_topics]

        # Now the multi-class queue where we consider a different mu per-topic
        lam_s_thru = lambdas.switch_thru
        mu_s_out = mus.switch_out
        denom = (1.0 - sum(lam/mu for lam, mu in zip(lam_s_thru, mu_s_out)))
        delta_s_out = [(1.0/mu)/denom for mu in mu_s_out]

        final_delays = [delta_b_in, delta_b_out, delta_s_in, delta_s_out]
        final_delays = zip(*final_delays)
        final_delays = [sum(terms) for terms in final_delays]

        # Only topics for which the subscriber will actually receive events should have expected service delays!
        # A subscriber receives events for a topic if they're subscribed to it AND it actually makes its way to the subscriber
        subs = set(configuration.subscriptions)
        final_lambdas = lambdas.sub_in
        final_delays = [d if (t in subs and l > 0.0) else 0.0 for t, d, l in zip(configuration.topics, final_delays, final_lambdas)]

        return final_delays

    def arrival_rates(self, configuration, subscriber=None):
        """
        Returns the expected arrival rates at each queue of all topics for the optionally-specified subscriber.
        :param configuration:
        :param subscriber:
        :return:
        :rtype: FiredexAlgorithm.Lambdas[list[float]]
        """

        # ENHANCE: consider publisher queues?

        # First, consider the broker queues:
        lambdas_b_in = self.broker_arrival_rates(configuration)

        # we simply 0 out any topics for which no subscriptions
        # TODO: handle multiple subscribers!
        # NOTE: to do this, we'll have to potentially consider other subscribers anyway as a queue shared by two subs
        # still has arrival rates for topics to which one of those subscribers is not interested.
        subs = set(configuration.subscriptions)
        lambdas_b_thru = [l if t in subs else 0.0 for t, l in zip(configuration.topics, lambdas_b_in)]
        lambdas_b_out = lambdas_b_thru
        # ENHANCE: per-flow lambdas?

        # Next, the SDN switch queues:
        # consider drop rate en route to switch
        net_flows = self.get_topic_net_flows(configuration, subscriber)
        lambdas_s_in = [l * (1.0 - self.get_drop_rates(configuration, subscriber)[net_flows[topic]])\
                        for l, topic in zip(lambdas_b_out, configuration.topics)]
        # ENHANCE: consider finite buffer size in prioq?
        lambdas_s_thru = lambdas_s_in
        lambdas_s_out = lambdas_s_thru

        # Lastly, the arrival rate at the subscriber consider packet errors
        lambdas_delivery = [(1 - configuration.error_rate) * l for l in lambdas_s_out]

        return FiredexAlgorithm.Lambdas(lambdas_b_in, lambdas_b_thru, lambdas_b_out,
                                        lambdas_s_in, lambdas_s_thru, lambdas_s_out, lambdas_delivery)

    def delivery_rates(self, configuration, subscriber=None):
        """
        Returns the expected delivery rates of all topics for the optionally-specified subscriber.
        :param configuration:
        :param subscriber:
        :return:
        :rtype: list[float]
        """
        return self.arrival_rates(configuration, subscriber=subscriber).sub_in

    def broker_arrival_rates(self, configuration):
        """
        Calculates the arrival rate of each topic at the broker by taking into account how many publishers advertise
        each topic i.e. arrival_rate[i] = pub_rate[i] * npubs_on_topic_i
        Note that no publishers on this topic results in a 0.0 arrival rate.
        :param configuration:
        :type configuration: FiredexConfiguration
        :return:
        :rtype: list[float]
        """

        # Start at 0 (no pubs) and fill in for each publishers' topics
        lambdas = {top: 0.0 for top in configuration.topics}
        for pub_class_ads in configuration.advertisements:
            for pub_ads in pub_class_ads.values():
                for topic in pub_ads:
                    lambdas[topic] += configuration.pub_rates[topic]
        lambdas = lambdas.items()
        lambdas.sort()
        lambdas = [v for (k,v) in lambdas]
        return lambdas
    # ENHANCE: consider publisher-to-subscriber error rates in above function
    publication_rates = broker_arrival_rates

    def ros_okay(self, configuration, tolerance=None):
        """
        Verifies if the "ro" condition is satisfied: whether the queues will have bounded sizes and not saturate over time.
        :param configuration:
        :param tolerance: configures this check such that for all queues, sum(ros) < (1.0 - tolerance) (default=self.ro_tolerance)
        :return: True if condition satisfied, False otherwise
        """
        if tolerance is None:
            tolerance = self.ro_tolerance

        ros = self.get_ros(configuration)
        ros_okay = all(sum(qros) < (1.0 - tolerance) for qros in ros)
        return ros_okay

    def get_ros(self, configuration):
        """
        Verifies if the "ro" condition is satisfied: whether the queues will have bounded sizes and not saturate over time.
        :param configuration:
        :return: list of the queues being considered with each entry being the ro values for each topic at that queue
        """

        lambdas = self.arrival_rates(configuration)
        mus = self.service_rates(configuration)

        # mash them up correctly
        all_queues_lam_mus = [(lambdas.broker_in, mus.broker_in), (lambdas.broker_thru, mus.broker_out),
                              (lambdas.switch_in, mus.switch_in), (lambdas.switch_thru, mus.switch_out)]

        ros = [[lam / mu for lam, mu in zip(topics_lams, topics_mus)] for topics_lams, topics_mus in all_queues_lam_mus]
        # log.info("ROs: %s" % ros)
        return ros

    ### Utility functions

    def estimate_utilities(self, configuration):
        """
        Estimate the expected utility for all subscriptions given the configuration.
        :param configuration:
        :type configuration: FiredexConfiguration
        :return: the estimated utilities, which matches the return structure of configuration.subscriptions
        """

        subs = configuration.subscriptions
        util_weights = configuration._utility_weights

        # need to convert these lists of size ntopics to size nsubs
        delays = self.total_delays(configuration)  # per subscriber?
        delays = [delays[s] for s in subs]
        lambdas = self.delivery_rates(configuration)  # per subscriber?
        lambdas = [lambdas[s] for s in subs]
        max_lambdas = self.publication_rates(configuration)
        max_lambdas = [max_lambdas[s] for s in subs]

        return [self.calculate_utility(dr, mdr, d, w) for dr, mdr, d, w in zip(lambdas, max_lambdas, delays, util_weights)]

    def calculate_utility(self, delivery_rate, max_delivery_rate, delay, weight):
        """
        Calculates the utility for a particular subscription according to the specified parameters, which may be
        estimates or actual measured values.
        :param delivery_rate: rate of successful notification delivery to the subscriber
        :param max_delivery_rate: rate of original publications that match this subscription
        :param delay: end-to-end delay from publication to subscriber reception (in seconds)
        :param weight:
        :return: a non-negative number
        """
        return calculate_utility(delivery_rate, max_delivery_rate, delay, weight)

    ### Priority setting functions

    # TODO: consolidate_subscriber_flows=True as a param to the setters should ignore the specified subscriber and
    # set the flows/priorities for specified topics the same across all subscribers!  This might be used in a real
    # implementation in order to limit the number of OpenFlow flow rules used for prioritization

    def set_topic_net_flow(self, topic, net_flow, configuration, subscriber=None):
        """
        Set the network flow to be used for the given topic when forwarded to the specified subscriber.  Not specifying
        subscriber sets this flow for ALL subscribers in the given configuration.
        :param topic:
        :param net_flow:
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: defaults to all subscribers
        :return:
        """

        if subscriber is None:
            subscribers = configuration.subscribers
        else:
            subscribers = [subscriber]

        for sub in subscribers:
            self._topic_flow_map.setdefault(configuration, dict()).setdefault(sub, dict())[topic] = net_flow

    def set_net_flow_priority(self, net_flow, priority, configuration, subscriber=None):
        """
        Set the priority class for the given network flow.
        :param net_flow:
        :param priority:
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: defaults to all subscribers
        :return:
        """

        if subscriber is None:
            subscribers = configuration.subscribers
        else:
            subscribers = [subscriber]

        for sub in subscribers:
            self._flow_prio_map.setdefault(configuration, dict()).setdefault(sub, dict())[net_flow] = priority

    def set_net_flow_drop_rate(self, net_flow, drop_rate, configuration, subscriber=None):
        """
        Set the preemptive drop rate for the given network flow.
        :param net_flow:
        :param drop_rate: should be in range [0,1]
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: defaults to all subscribers
        :return:
        """

        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError("requested drop_rate (%f) not in expected range of [0,1]")

        if subscriber is None:
            subscribers = configuration.subscribers
        else:
            subscribers = [subscriber]

        for sub in subscribers:
            self._drop_rates.setdefault(configuration, dict()).setdefault(sub, dict())[net_flow] = drop_rate

    def get_drop_rates(self, configuration, subscriber=None):
        """
        Returns the drop rates for the requested subscriber's network flows.

        NOTE: this assumes the algorithm has already been run!

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: which subscriber's drop rates are being requested (default=arbitrarily pick one, which assumes
            the values were set across all subscribers)
        :return: mapping of network flows to drop rates
        :rtype: dict
        """

        try:
            if subscriber is None:
                # arbitrary subscriber from underlying mapping
                return next(self._drop_rates[configuration].itervalues())
            else:
                return self._drop_rates[configuration][subscriber].copy()
        except IndexError as e:
            raise ValueError("request for bad configuration or subscriber caused error: %s" % e)

    def get_topic_priorities(self, configuration, subscriber=None):
        """
        Runs the actual algorithm to determine what the priority levels should be according to the current real-time
        configuration specified.  This implementation just defers to the get_topic_net_flows() and
        get_net_flow_priorities() methods.

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: which subscriber priorities being requested (default=arbitrarily pick one, which assumes
            the values were set across all subscribers)
        :return: mapping of topic IDs to priority classes
        :rtype: dict
        """

        if self._update_needed(configuration, subscriber):
            self.__run_algorithm(configuration, subscriber)

        topic_flow_map = self.get_topic_net_flows(configuration, subscriber)
        flow_prio_map = self.get_net_flow_priorities(configuration, subscriber)

        topic_prio_map = {t: flow_prio_map[f] for t, f in topic_flow_map.items()}
        return topic_prio_map

    def get_topic_net_flows(self, configuration, subscriber=None):
        """
        Runs the algorithm to assign topics to network flows based on current configuration state.
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: which subscriber net flows being requested (default=arbitrarily pick one, which assumes
            the values were set across all subscribers)
        :return: mapping of topic IDs to network flow IDs
        :rtype: dict
        """

        if self._update_needed(configuration, subscriber):
            self.__run_algorithm(configuration, subscriber)
        if subscriber is None:
            # arbitrary subscriber from underlying mapping
            return next(self._topic_flow_map[configuration].itervalues())
        else:
            return self._topic_flow_map[configuration][subscriber]

    def get_net_flow_priorities(self, configuration, subscriber=None):
        """
        Runs the algorithm to assign network flows to priority levels based on current configuration state.
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: which subscriber priorities being requested (default=arbitrarily pick one, which assumes
            the values were set across all subscribers)
        :return: mapping of network flow IDs to priority classes
        :rtype: dict
        """

        if self._update_needed(configuration, subscriber):
            self.__run_algorithm(configuration, subscriber)
        if subscriber is None:
            # arbitrary subscriber from underlying mapping
            return next(self._flow_prio_map[configuration].itervalues())
        else:
            return self._flow_prio_map[configuration][subscriber]

    def force_update(self, configuration=None, subscribers=None):
        """
        Forces the algorithm to update the management (e.g. priorities) for certain configurations/subscribers.
        :param configuration: which configuration should be updated (default=all configs, which ignores subscribers arg)
        :type configuration: FiredexConfiguration
        :param subscribers: which subscribers for that configuration should be updated (default=all subscribers)
        """

        # to blow away all configurations, just reset everything
        if configuration is None:
            self._topic_flow_map = dict()
            self._flow_prio_map = dict()

        # for a specific configuration, just delete it IF IT EXISTS
        elif subscribers is None:
            self._topic_flow_map.pop(configuration, None)
            self._flow_prio_map.pop(configuration, None)

        # otherwise, carefully pop off each subscriber for the configuration (again, IF IT EXISTS)
        else:
            for subscriber in subscribers:
                self._topic_flow_map.get(configuration, dict()).pop(subscriber, None)
                self._flow_prio_map.get(configuration, dict()).pop(subscriber, None)

    def __run_algorithm(self, configuration, subscribers=None):
        """
        DO NOT override this method unless you implement your own preemptive drop rate policy!
        Runs the algorithm to assign network flows to priority levels based on current configuration state.  Also runs
        the preemptive drop rate policy afterwards to ensure ro conditions will be met i.e. queues should be stable.

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscribers: which subscribers should have their priority levels assigned (default=None=all subscribers)
        """

        self._run_algorithm(configuration, subscribers)
        self._apply_drop_rate_policy(configuration, subscribers)

    ### Override these as necessary in derived algorithm classes

    def _run_algorithm(self, configuration, subscribers=None):
        """
        Runs the algorithm to assign network flows to priority levels based on current configuration state.
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscribers: which subscribers should have their priority levels assigned (default=None=all subscribers)
        """

        # NOTE: do this in derived class implementations to ensure you set the priorities for all subscribers!
        if subscribers is None:
            subscribers = configuration.subscribers

        raise NotImplementedError

    def _update_needed(self, configuration, subscriber=None):
        """
        Determines if an update is needed for the given configuration.  If the algorithm hasn't been run yet, this
        should return True.  This default base class implementation returns True if this config/sub has not been
        seen yet.  Hence, the calling class should precede this call with one to force_update(configuration) if the
        configuration has changed or enough time has passed! Base classes should probably override this method
        especially for actual system implementations to consider e.g. overhead of updating priorities.

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: which subscriber may need updates (default=all subscribers)
        :return: True if the caller should do _run_algorithm(...) before proceeding
        :rtype: bool
        """

        # when we're checking all subscribers, need to make sure they're ALL present!
        if subscriber is None:
            subs = configuration.subscribers
        else:
            subs = [subscriber]

        update = configuration not in self._topic_flow_map or any(s not in self._topic_flow_map[configuration] for s in subs)
        return update

    def _apply_drop_rate_policy(self, configuration, subscribers=None, policy=None):
        """
        Applies the specified drop rate policy (or default if unspecified) to ensure the ro conditions of the queue
        model will be met, which should result in the queues being stable (not growing without bound).

        NOTE: this default implementation assumes the priorities have already been set and so, depending on the policy
              requested, it basically keeps raising the drop probability for all network flows until the ro conditions
              are met.

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscribers: which subscribers should have their drop rates assigned (default=None=all subscribers)
        :raises ValueError: if an iterative policy hits its max # attempts to meet ros or an unknown policy is requested
        :return:
        """

        if subscribers is None:
            subscribers = configuration.subscribers

        if policy is None:
            policy = self.drop_policy

        # This basic policy sets drop rates for each net flow according to its assigned priority level where the
        # drop rate = 1- x^(-prio-1), where x starts at 1.0 and increases slightly until the ro conditions are met
        if policy == 'expon':
            exp_base = 1.0
            ros_met = False
            iterations_left = 1000
            while not ros_met and iterations_left > 0:
                for sub in subscribers:
                    for net_flow, prio in self.get_net_flow_priorities(configuration, sub).items():
                        drop_rate = 1.0 - exp_base**(-prio-1)
                        self.set_net_flow_drop_rate(net_flow, drop_rate, configuration, sub)

                ros_met = self.ros_okay(configuration)
                exp_base += 0.1
                iterations_left -= 1

            if iterations_left == 0:
                raise QueueStabilityError("Max iterations for drop rate policy 'expon' reached!  Check model constraints! "
                                 "Drop rates ended up being: %s" % self.get_drop_rates(configuration))
        else:
            raise QueueStabilityError("unrecognized preemptive drop rate policy %s" % policy)
