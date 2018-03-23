from firedex_algorithm import FiredexAlgorithm

import logging
log = logging.getLogger(__name__)


class NullFiredexAlgorithm(FiredexAlgorithm):
    """
    Algorithm implementation that assigns the same priority for each subscription as a baseline comparison with actual algorithms.
    """

    def __init__(self, **kwargs):
        super(NullFiredexAlgorithm, self).__init__(**kwargs)

    def _run_algorithm(self, configuration, subscribers=None):
        if subscribers is None:
            subscribers = configuration.subscribers

        for sub in subscribers:
            flows = configuration.net_flows_for_subscriber(sub)
            flow = flows[0]
            prio = configuration.prio_classes[0]

            for req in configuration.get_subscriptions(sub):
                self.set_subscription_net_flow(req, flow, configuration)
            for f in flows:
                self.set_net_flow_priority(f, prio, configuration)
