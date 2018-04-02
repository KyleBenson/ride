from scale_client.sensors.virtual_sensor import VirtualSensor


class FiredexSubscriber(VirtualSensor):
    """
    FireDeX client-side middleware for the SCALE client.  Enables subscribers to subscribe to certain event topics on
    different network connections, which allows SDN management of each flow independently.
    """

    def __init__(self, broker, net_flows=tuple(), static_topic_flow_map=None, subscriptions=tuple(), **kwargs):
        """
        :param broker:
        :param net_flows: list of network flow objects this VS should configure
        :type net_flows: list|tuple
        :param static_topic_flow_map: static mapping of subscription topics to network flows (i.e. net flow indices)
        :type static_topic_flow_map: dict
        :param subscriptions:
        :param kwargs:
        """

        # XXX: we don't want to pass subscriptions along or the Application will subscribe to them internally!
        super(FiredexSubscriber, self).__init__(broker, subscriptions=tuple(), **kwargs)

        self._net_flows = net_flows
        self._static_topic_flow_map = static_topic_flow_map

        if static_topic_flow_map and not net_flows:
            raise ValueError("specified a static_topic_flow_map without any network flows!")

    @property
    def remote_addresses(self):
        """
        Returns the addresses (i.e. a tuple of e.g. ipv4, port) for each requested network flow.
        :return:
        """
        return self._net_flows

    def address_for_topic(self, topic):
        """
        Returns the address (i.e. a tuple of e.g. ipv4, port) for the network flow the requested topic is assigned to.
        :param topic:
        :return:
        """

        if not self._static_topic_flow_map:
            raise ValueError("no static_topic_flow_map specified but topic %s's address was requested!" % topic)

        try:
            flow_idx = self._static_topic_flow_map[topic]
        except IndexError:
            raise ValueError("no static_topic_flow_map entry found for topic %s's address!" % topic)

        try:
            return self.remote_addresses[flow_idx]
        except IndexError:
            raise ValueError("no net_flows entry found for topic %s's flow index %d!" % (topic, flow_idx))
