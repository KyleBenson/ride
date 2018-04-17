from scale_client.sensors.network.coap_sensor import CoapSensor
from scifire.scale.firedex_subscriber import FiredexSubscriber

import logging
log = logging.getLogger(__name__)


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
        self._subs = subscriptions
        self._remote_path = remote_path

    # XXX: do this in on_start so we can e.g. delay it
    def on_start(self):
        super(FiredexCoapSubscriber, self).on_start()

        # we need to gather up the subscriptions for each CoapSensor and create a coap client for each network flow
        sensor_configs = {f: [] for f in self._net_flows}

        for sub in self._subs:
            flow = self.address_for_topic(sub)
            sensor_configs[flow].append(sub)

        for flow, subs in sensor_configs.items():
            kwargs = dict()
            if flow.src_port:
                kwargs['src_port'] = flow.src_port
            if flow.dst_port:
                kwargs['port'] = flow.dst_port
            if flow.dst_addr:
                kwargs['hostname'] = flow.dst_addr

            # XXX: set timeout to be shorter so we will re-attempt to observe if the first try failed
            timeout = 20

            subs = [self._remote_path % sub for sub in subs]
            client = CoapSensor(self._broker, subscriptions=subs, timeout=timeout, **kwargs)
            client.on_start()
            self._clients.append(client)
            log.debug("FiredexCoapSubscriber added CoapSensor(sub_port=%s) with subs: %s" % (flow.src_port, str(subs)))
