# @author: Kyle Benson
# (c) Kyle Benson 2018

import argparse

from scifire.config import *


class FiredexScenario(object):
    """
    Basic representation of the Sci-Fire data exchange scenario.  Used by other classes e.g. experiments, algorithms...
    """

    def __init__(self, num_ffs=DEFAULT_NUM_FFS, num_iots=DEFAULT_NUM_IOTS,
                 num_net_flows=DEFAULT_NUM_NET_FLOWS, num_priority_levels=DEFAULT_NUM_PRIORITIES,
                 num_topics=DEFAULT_NUM_TOPICS,
                 # XXX: for multiple inheritance
                 **kwargs):
        # XXX: ensure we call all the super constructors!
        try:
            super(FiredexScenario, self).__init__(**kwargs)
        except TypeError:
            super(FiredexScenario, self).__init__()

        self.num_ffs = num_ffs
        self.num_iots = num_iots
        self.num_net_flows = num_net_flows
        self.num_priority_levels = num_priority_levels
        self.num_topics = num_topics

    @classmethod
    def get_arg_parser(cls, parents=(), add_help=False):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        arg_parser = argparse.ArgumentParser(parents=parents, add_help=add_help, conflict_handler='resolve')

        # experimental treatment parameters
        arg_parser.add_argument('--num-ffs', '-nf', dest='num_ffs', type=int, default=DEFAULT_NUM_FFS,
                                help='''The number of fire fighter 'hosts' to create, which represent a FF equipped with
                                IoT devices that relay their data through some wireless smart hub (default=%(default)s).''')
        arg_parser.add_argument('--num-iots', '-nd', '-ni', dest='num_iots', type=int, default=DEFAULT_NUM_IOTS,
                                help='''The number of IoT device hosts to create, which represent various sensors,
                                actuators, or other IoT devices that reside within the building and publish
                                fire event-related data to the BMS (default=%(default)s).''')
        arg_parser.add_argument('--num-net-flows', '-nn', dest='num_net_flows', type=int, default=DEFAULT_NUM_NET_FLOWS,
                                help='''The number of distinct network flows (per host) to consider, which will
                                correspond to the number of distinct pub-sub client connections to open up.  Each flow
                                is mapped to a priority level to prioritize the topics transmitted on that connection
                                (default=%(default)s).''')
        arg_parser.add_argument('--num-priorities', '-np', '-nq', dest='num_priority_levels', type=int, default=DEFAULT_NUM_PRIORITIES,
                                help='''The number of priority levels to consider, which will correspond to the number
                                 of priority queues configured in the network (default=%(default)s).''')
        arg_parser.add_argument('--num-topics', '-nt', dest='num_topics', type=int, default=DEFAULT_NUM_TOPICS,
                                help='''The number of event topics to consider, which will be mapped to network flows
                                 for prioritization by the network (default=%(default)s).''')

        # TODO: topic-generation models: who is interested? what's their utility? data rate needed? periodic vs. async?
        # TODO: utility function?
        # TODO: event-generation models (data requests, network changes, and sensed events) and random seeds?
        # TODO: how to bring in network-related parameters?  Maybe make a NetworkScenario base class?
        # TODO: some notion of time?

        return arg_parser
