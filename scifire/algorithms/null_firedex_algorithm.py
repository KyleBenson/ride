from firedex_algorithm import FiredexAlgorithm

import logging
log = logging.getLogger(__name__)


class NullFiredexAlgorithm(FiredexAlgorithm):
    """
    Algorithm implementation that assigns priorites randomly for a baseline comparison with actual algorithms.
    """

    def __init__(self, **kwargs):
        super(NullFiredexAlgorithm, self).__init__(**kwargs)

    def _run_algorithm(self, configuration, subscribers=None):
        flow = configuration.net_flows[0]
        prio = configuration.prio_classes[0]

        # NOTE: we just ignore the subscribers parameter here since all flows/priorities will be the same!
        # TODO: when we update the config.net_flows API for multiple subscribers, we'll have to change this a bit...

        for t in configuration.topics:
            self.set_topic_net_flow(t, flow, configuration)
        for f in configuration.net_flows:
            self.set_net_flow_priority(f, prio, configuration)
