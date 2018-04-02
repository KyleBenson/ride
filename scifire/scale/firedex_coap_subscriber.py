from scale_client.sensors.network.coap_sensor import CoapSensor
from scifire.scale.firedex_subscriber import FiredexSubscriber


class FiredexCoapSubscriber(FiredexSubscriber):
    """
    FireDeX client-side middleware for the SCALE client that supports CoAP-based SDN-prioritized subscriptions.
    """

    def __init__(self, broker, remote_path="/events/%s", subscriptions=tuple(), **kwargs):
        """
        Creates a CoapSensor for each requested subscription and configures it for the corresponding
        network flow (connection).
        :param remote_path: the remote URI path that the subscription topic will be filled into for creating final URIs
        """

        super(FiredexCoapSubscriber, self).__init__(broker, subscriptions=subscriptions, **kwargs)

        # TODO: how to handle dynamic assignments???

        self._clients = []
        for sub in subscriptions:
            flow = self.address_for_topic(sub)
            kwargs = dict()
            if flow.src_port:
                kwargs['src_port'] = flow.src_port
            if flow.dst_port:
                kwargs['port'] = flow.dst_port
            if flow.dst_addr:
                kwargs['hostname'] = flow.dst_addr
            self._clients.append(CoapSensor(broker, topic=remote_path % sub, **kwargs))
