import random

from firedex_algorithm import FiredexAlgorithm

import logging
log = logging.getLogger(__name__)


class RandomFiredexAlgorithm(FiredexAlgorithm):
    """
    Algorithm implementation that assigns priorities randomly (uniform dist. across all priority classes) for a
    baseline comparison with actual algorithms.
    """

    def __init__(self, seed=None, **kwargs):
        super(RandomFiredexAlgorithm, self).__init__(**kwargs)
        self.rand = random.Random(seed)

    # TODO: always return lowest prio level for non-subscribed topics?
    def _run_algorithm(self, configuration, subscribers=None):
        if subscribers is None:
            subscribers = configuration.subscribers

        flows = configuration.net_flows
        for t in configuration.topics:
            flow = self.rand.sample(flows, 1)[0]
            for sub in subscribers:
                self.set_topic_net_flow(t, flow, configuration, subscriber=sub)

        # XXX: because we currently assume nflows == nprios, we need to make sure each prioq is used here or we end up
        # with e.g. only 2/3 priorities used!  Hence, for now we just assign flows to directly map to their
        # corresponding priority: in the future we should start by sampling priorities for net flows to ensure each
        # prio is used, and then re-sample until all remaining net flows are used up.

        prios = configuration.prio_classes
        if len(flows) != len(prios):
            raise NotImplementedError("no support yet for randomly assigning net flow -> priorities!  We're just"
                                      "assuming that #flows == #prio classes!")

        for f, p in zip(flows, prios):
            for sub in subscribers:
                self.set_net_flow_priority(f, p, configuration, subscriber=sub)
