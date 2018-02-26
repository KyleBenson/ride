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
    def _run_algorithm(self, configuration):
        flows = configuration.net_flows
        prios = configuration.prio_classes
        for t in configuration.topics:
            self.set_topic_net_flow(t, self.rand.sample(flows, 1)[0])
        for f in configuration.net_flows:
            self.set_net_flow_priority(f, self.rand.sample(prios, 1)[0])
