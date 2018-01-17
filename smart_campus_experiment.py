CLASS_DESCRIPTION = '''Experiment that models failures in a smart campus network setting
and determines the effectiveness of the Resilient IoT Data Exchange (RIDE) middleware.
RIDE improves data collection using RIDE-C, which routes IoT data publishers to the cloud
server through the best available DataPath or directly to the edge server if they aren't available.
RIDE improves alert dissemination using RIDE-D, which utilizes SDN-enabled IP multicast and several
multicast tree-constructing and choosing algorithms for creation of
 Multiple Maximally-Disjoint Multicast Trees (MDMTs).'''

# @author: Kyle Benson
# (c) Kyle Benson 2017

import os
import argparse
import logging
log = logging.getLogger(__name__)

import networkx as nx
import ride
from ride.config import *
from network_experiment import NetworkExperiment
from failure_model import SmartCampusFailureModel
DISTANCE_METRIC = 'latency'  # for shortest path calculations

class SmartCampusExperiment(NetworkExperiment):

    def __init__(self, nsubscribers=5, npublishers=5, failure_model=None, topology_filename=None,
                 ## RideD params
                 ntrees=4, tree_construction_algorithm=DEFAULT_TREE_CONSTRUCTION_ALGORITHM,
                 ## RideC params
                 reroute_policy=DEFAULT_REROUTE_POLICY,
                 # flags to enable/disable certain features for running different combinations of experiments
                 with_ride_d=True, with_ride_c=True,
                 # HACK: kwargs just used for construction via argparse since they'll include kwargs for other classes
                 **kwargs):
        """
        :param ntrees:
        :param tree_construction_algorithm:
        :param nsubscribers:
        :param npublishers:
        :param reroute_policy: used by RideC to determine the publishers' routes to edge server (after re-route from cloud in our real scenario)
        :param failure_model:
        :param topology_filename:
        :param with_ride_d:
        :param with_ride_c:
        :param kwargs:
        """
        super(SmartCampusExperiment, self).__init__()

        self.ntrees = ntrees
        self.nsubscribers = nsubscribers
        self.npublishers = npublishers

        self.topology_filename = topology_filename
        self.topo = None  # built later in setup_topology()

        self.tree_construction_algorithm = tree_construction_algorithm
        self.reroute_policy = reroute_policy

        self.with_ride_c = with_ride_c
        self.with_ride_d = with_ride_d
        self.with_cloud = with_ride_c

        if failure_model is None:
            failure_model = SmartCampusFailureModel()
        self.failure_model = failure_model

        # These fields will be set for each run so as to avoid passing them around as parameters.
        self.subscribers = None
        self.publishers = None
        self.edge_server = None
        self.cloud = None
        self.failed_nodes, self.failed_links = (None, None)

        self.results['params'].update({'ntrees': ntrees,
                                   'nsubscribers': nsubscribers,
                                   'npublishers': npublishers,
                                   'failure_model': self.failure_model.get_params(),
                                   'failrandseed': kwargs.get('failure_rand_seed', None),
                                   'heuristic': self.get_mcast_heuristic_name(),
                                   'reroute_policy': self.reroute_policy,
                                   'topo': topology_filename,})

    @classmethod
    def get_arg_parser(cls, parents=(SmartCampusFailureModel.arg_parser, NetworkExperiment.get_arg_parser(),
                                     ride.ride_d.RideD.get_arg_parser()),
                       add_help=False):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        # TODO: add RideC parameters?

        arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION,
                                             parents=parents, add_help=add_help)
        # experimental treatment parameters
        arg_parser.add_argument('--nsubscribers', '-s', type=int, default=5,
                            help='''number of multicast subscribers (terminals) to reach (default=%(default)s)''')
        arg_parser.add_argument('--npublishers', '-p', type=int, default=5,
                            help='''number of IoT sensor publishers to contact edge server (default=%(default)s)''')
        arg_parser.add_argument('--topology-filename', '--topo', type=str, default='topos/campus_topo.json', dest='topology_filename',
                            help='''file name of topology to use (default=%(default)s)''')
        arg_parser.add_argument('--reroute-policy', '--route', type=str, default=DEFAULT_REROUTE_POLICY, dest='reroute_policy',
                            help='''policy for (re)routing publishers to the edge server (default=%(default)s)''')

        return arg_parser

    # ENHANCE: maybe a version that uses the members rather than being classmethod?
    @classmethod
    def build_default_results_file_name(cls, args, dirname='results'):
        """
        :param args: argparse object (or plain dict) with all args info (not specifying ALL args is okay)
        :param dirname: directory name to place the results files in
        :return: string representing the output_filename containing a parameter summary for easy identification
        """

        # Convert argparse object to dict
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Pass empty args to get the default configurations.
        defaults = cls.get_arg_parser().parse_args(args=[])

        # Extract topology file name
        try:
            topo_fname = args.get('topology_filename', defaults.topology_filename)
            topo_fname = os.path.splitext(os.path.basename(topo_fname).split('_')[2])[0]
        except IndexError:
            # topo_fname must not be formatted as expected: just use it plain but remove _'s to avoid confusing code parsing the topo for its params
            topo_fname = os.path.splitext(os.path.basename(args.get('topology_filename', defaults.topology_filename).replace('_', '')))[0]

        output_filename = 'results_%dt_%0.2ff_%ds_%dp_%s_%s_%s_%0.2fe.json' % \
                          (args.get('ntrees', defaults.ntrees), args.get('fprob', defaults.fprob),
                           args.get('nsubscribers', defaults.nsubscribers), args.get('npublishers', defaults.npublishers),
                           cls.build_mcast_heuristic_name(*args.get('tree_construction_algorithm', defaults.tree_construction_algorithm)),
                           # not currently configurable via cmd line...
                           args.get('reroute_policy', DEFAULT_REROUTE_POLICY),
                           topo_fname, args.get('error_rate', defaults.error_rate))

        output_filename = os.path.join(dirname, output_filename)

        return output_filename

    @classmethod
    def build_from_args(cls, args):
        """Constructs from command line arguments."""

        args = cls.get_arg_parser().parse_args(args)

        # convert to plain dict
        args = vars(args)
        failure_model = SmartCampusFailureModel(**args)
        args['failure_model'] = failure_model

        if args['output_filename'] is None:
            args['output_filename'] = cls.build_default_results_file_name(args)

        return cls(**args)

    def record_result(self, result):
        # First, add additional parameters used on this run.
        result['failed_nodes'] = self.failed_nodes
        result['failed_links'] = self.failed_links
        return super(SmartCampusExperiment, self).record_result(result)

    def get_failed_nodes_links(self):
        """Returns which nodes/links failed according to the failure model.
        @:return failed_nodes, failed_links"""
        nodes, links = self.failure_model.apply_failure_model(self.topo)
        log.debug("Failed nodes: %s" % nodes)
        log.debug("Failed links: %s" % links)
        return nodes, links

    def choose_subscribers(self):
        # ENHANCE: could sample ALL of the hosts and then just slice off nsubs.
        # This would make it so that different processes with different nhosts
        # but same topology would give complete overlap (smaller nsubs would be
        # a subset of larger nsubs).  This would still cause significant variance
        # though since which hosts are chosen is different and that affects outcomes.
        subs = self._choose_random_hosts(self.nsubscribers)
        log.debug("Subscribers: %s" % subs)
        return subs

    def choose_publishers(self):
        pubs = self._choose_random_hosts(self.npublishers)
        log.debug("Publishers: %s" % pubs)
        return pubs

    def choose_server(self):
        server = self.random.choice(self.topo.get_servers())
        log.debug("Server: %s" % server)
        return server

    def setup_experiment(self):
        """
        Set up the experiment and configure it as necessary before run_experiment is called.
        By default it chooses the subscribers, publishers, server, and failed nodes/links
        for this experimental run.

        :return:
        """
        self.subscribers = self.choose_subscribers()
        self.publishers = self.choose_publishers()
        # NOTE: this is unnecessary as we only have a single server in our test topos.  If we use multiple, need
        # to actually modify RideD here with updated server.
        self.edge_server = self.choose_server()
        self.failed_nodes, self.failed_links = self.get_failed_nodes_links()

        assert self.edge_server not in self.failed_nodes, "shouldn't be failing the server!  useless run...."

    @staticmethod
    def build_mcast_heuristic_name(*args):
        """The heuristic is given with arguments so we use this function
        to convert it to a compact human-readable form.  This is a
        separate static function for use by other classes."""
        if len(args) > 1:
            interior = ",".join(args[1:])
            return "%s[%s]" % (args[0], interior)
        else:
            return args[0]

    def get_mcast_heuristic_name(self):
        return self.build_mcast_heuristic_name(*self.tree_construction_algorithm)

    ### Helper functions for working with failures and calculating reachability in our topology

    @staticmethod
    def get_oracle_reachability(subscribers, server, failed_topology):
        """Gets the reachability for the 'oracle' heuristic, which is simply calculating how many subscribers are
        reachable by ANY path"""
        failed_topology.graph['heuristic'] = 'oracle'
        topos_to_check = [failed_topology]
        reach = SmartCampusExperiment.get_reachability(server, subscribers, topos_to_check)[0]
        return reach

    @staticmethod
    def get_reachability(server, subscribers, topologies):
        """Returns the average probability of reaching any of the subscribers from the
        server in each of the given topologies.  Also includes the result for using
        all topologies at once.
        :returns list: containing reachability for each topology (in order),
        with the last entry representing using all topologies at the same time
        """

        subscribers = set(subscribers)
        subs_reachable_by_tree = []
        all_subscribers_reachable = set()
        for topology in topologies:
            nodes_reachable = set(nx.single_source_shortest_path(topology, server))
            # could also use has_path()
            subscribers_reachable = subscribers.intersection(nodes_reachable)
            subs_reachable_by_tree.append(len(subscribers_reachable) / float(len(subscribers)))
            all_subscribers_reachable.update(subscribers_reachable)

            log.debug("%s heuristic reached %d subscribers in this topo" % (topology.graph['heuristic'], len(subscribers_reachable)))
        log.debug("ALL subscribers reached by these topos: %d" % len(all_subscribers_reachable))
        # Lastly, include all of them reachable
        subs_reachable_by_tree.append(float(len(all_subscribers_reachable)) / len(subscribers))
        return subs_reachable_by_tree

    @staticmethod
    def get_failed_topology(topo, failed_nodes, failed_links):
        """Returns a copy of the graph topo with all of the failed nodes
        and links removed."""
        # since we froze the graph we can't just use .copy()
        topology = nx.Graph(topo)
        topology.remove_edges_from(failed_links)
        topology.remove_nodes_from(failed_nodes)
        return topology