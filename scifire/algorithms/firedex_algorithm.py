from ..firedex_configuration import FiredexConfiguration

import logging
log = logging.getLogger(__name__)


class FiredexAlgorithm(object):
    """
    Abstract base class for assigning topic priorities (i.e. topic-> net flow -> prio mappings) according to the current
    (estimated or theoretical) system configuration/state. Derived classes will share the same analytical model that
    uses our queuing network model, but will calculate the mappings differently.
    """

    def __init__(self, **kwargs):
        super(FiredexAlgorithm, self).__init__(**kwargs)

        # these are filled in by _run_algorithm()
        self._topic_flow_map = dict()
        self._flow_prio_map = dict()

    def set_topic_net_flow(self, topic, net_flow, subscriber=None):
        """
        Set the network flow to be used for the given topic when forwarded to the specified subscriber.  Not specifying
        subscriber sets this flow for ALL subscribers.
        :param topic:
        :param net_flow:
        :param subscriber: defaults to all subscribers
        :return:
        """

        # TODO: handle multiple (Specific) subscribers!
        if subscriber is not None:
            log.error("set_topic_net_flow for a specific subscriber not yet supported!")

        self._topic_flow_map[topic] = net_flow

    def set_net_flow_priority(self, net_flow, priority):
        """
        Set the priority class for the given network flow.
        :param net_flow:
        :param priority:
        :return:
        """

        self._flow_prio_map[net_flow] = priority

    # TODO: how to handle multiple subscribers???
    def get_topic_priorities(self, configuration):
        """
        Runs the actual algorithm to determine what the priority levels should be according to the current real-time
        configuration specified.  This implementation just defers to the get_topic_net_flows() and
        get_net_flow_priorities() methods.

        :param configuration:
        :type configuration: FiredexConfiguration
        :return: mapping of topic IDs to priority classes
        :rtype: dict
        """

        if self._update_needed(configuration):
            self._run_algorithm(configuration)

        topic_flow_map = self.get_topic_net_flows(configuration)
        flow_prio_map = self.get_net_flow_priorities(configuration)
        topic_prio_map = {t: flow_prio_map[f] for t, f in topic_flow_map.items()}
        return topic_prio_map

    def get_topic_net_flows(self, configuration):
        """
        Runs the algorithm to assign topics to network flows based on current configuration state.
        :param configuration:
        :type configuration: FiredexConfiguration
        :return: mapping of topic IDs to network flow IDs
        :rtype: dict
        """

        if self._update_needed(configuration):
            self._run_algorithm(configuration)
        return self._topic_flow_map

    def get_net_flow_priorities(self, configuration):
        """
        Runs the algorithm to assign network flows to priority levels based on current configuration state.
        :param configuration:
        :type configuration: FiredexConfiguration
        :return: mapping of network flow IDs to priority classes
        :rtype: dict
        """

        if self._update_needed(configuration):
            self._run_algorithm(configuration)
        return self._flow_prio_map

    def _run_algorithm(self, configuration):
        """
        Runs the algorithm to assign network flows to priority levels based on current configuration state.
        :param configuration:
        :type configuration: FiredexConfiguration
        """

        raise NotImplementedError

    def _update_needed(self, configuration):
        """
        Determines if an update is needed for the given configuration.  If the algorithm hasn't been run yet, this
        should return True.  This default base class implementation ALWAYS returns True so base classes should
        override it especially for actual system implementations.

        :param configuration:
        :type configuration: FiredexConfiguration
        :return: True if the caller should do _run_algorithm(...) before proceeding
        :rtype: bool
        """
        return True
