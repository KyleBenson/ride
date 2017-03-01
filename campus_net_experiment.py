#! /usr/bin/python

NEW_SCRIPT_DESCRIPTION = '''Experiment that models failures in a campus network setting
and determines the effectiveness of several SDN/IP multicast tree-establishing algorithms
in improving data dissemination to subscribing IoT devices around the campus.
Intended as a simplified simulation that removes the need for network setup or
components such as switches, SDN controllers, etc.'''

# @author: Kyle Benson
# (c) Kyle Benson 2017

from failure_model import SmartCampusFailureModel
# TODO: from sdn_topology import *
from networkx_sdn_topology import NetworkxSdnTopology
# from redundant_multicast_algorithms import ilp_redundant_multicast

import networkx as nx
import logging as log
import random
import json
import numpy as np
import argparse
import time
import signal

COST_METRIC = 'weight'  # for links only
DISTANCE_METRIC = 'latency'  # for shortest path calculations


class SmartCampusNetworkxExperiment(object):

    def __init__(self, nruns=1, ntrees=3, mcast_heuristic=('steiner',), nsubscribers=5, npublishers=5,
                 failure_model=None, topo=('networkx',),
                 debug='info', output_filename='results.json',
                 choice_rand_seed=None, rand_seed=None,
                 # NOTE: kwargs just used for construction via argparse
                 **kwargs):
        super(SmartCampusNetworkxExperiment, self).__init__()
        self.nruns = nruns
        self.ntrees = ntrees
        self.nsubscribers = nsubscribers
        self.npublishers = npublishers
        self.output_filename = output_filename
        self.mcast_heuristic = mcast_heuristic

        log_level = log.getLevelName(debug.upper())
        log.basicConfig(format='%(levelname)s:%(message)s', level=log_level)

        self.random = random.Random(choice_rand_seed)
        random.seed(rand_seed)
        # QUESTION: do we need one for the algorithms as well?

        if failure_model is None:
            failure_model = SmartCampusFailureModel()
        self.failure_model = failure_model

        if topo[0] == 'networkx':
            topology = NetworkxSdnTopology(*topo[1:])
        else:
            raise NotImplementedError("unrecognized or unimplemented SdnTopology type %s" % topo[0])
        self.topo = topology

        # results are output as JSON to file after the experiment runs
        self.results = {'results': [], # each is a single run containing: {run: run#, heuristic_name: percent_reachable}
                        'params': {'ntrees': ntrees,
                                   'nsubscribers': nsubscribers,
                                   'npublishers': npublishers,
                                   'failure_model': self.failure_model.get_params(),
                                   'heuristic': self.get_mcast_heuristic_name(),
                                   'topo': topo,
                                   'choicerandseed': choice_rand_seed,
                                   'randseed': rand_seed,
                                   'failrandseed': kwargs.get('failure_rand_seed', None),
                                   }
                        }

        # HACK for saving failure model's rand seed
        try:
            self.results['params']['failrandseed'] = kwargs['failure_rand_seed']
        except IndexError:
            pass

    @classmethod
    def build_from_args(cls, args):
        """Constructs from an ArgumentParser object."""
        ##################################################################################
        #################      ARGUMENTS       ###########################################
        # ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
        # action is one of: store[_const,_true,_false], append[_const], count
        # nargs is one of: N, ?(defaults to const when no args), *, +, argparse.REMAINDER
        # help supports %(var)s: help='default value is %(default)s'
        ##################################################################################

        parser = argparse.ArgumentParser(description=NEW_SCRIPT_DESCRIPTION,
                                         #formatter_class=argparse.RawTextHelpFormatter,
                                         #epilog='Text to display at the end of the help print',
                                         parents=[SmartCampusFailureModel.arg_parser]
                                         )

        # experimental treatment parameters
        parser.add_argument('--nruns', '-r', type=int, default=1,
                            help='''number of times to run experiment (default=%(default)s)''')
        parser.add_argument('--ntrees', '-t', type=int, default=4,
                            help='''number of redundant multicast trees to build (default=%(default)s)''')
        parser.add_argument('--mcast-heuristic', '-a', type=str, default=('steiner',), nargs='+', dest='mcast_heuristic',
                            help='''heuristic algorithm for building multicast trees.  First arg is the heuristic
                            name; all others are passed as args to the heuristic. (default=%(default)s)''')
        # TODO: should this be a percentage? should also warn if too many subs based on model of how many would need mcast support
        parser.add_argument('--nsubscribers', '-s', type=int, default=5,
                            help='''number of multicast subscribers (terminals) to reach (default=%(default)s)''')
        parser.add_argument('--npublishers', '-p', type=int, default=5,
                            help='''number of IoT sensor publishers to contact edge server (default=%(default)s)''')
        parser.add_argument('--topo', type=str, default=['networkx'], nargs='+',
                            help='''type of SdnTopology to use and (optionally) its constructor parameters (default=%(default)s)''')

        # experiment interaction control
        parser.add_argument('--debug', '-d', type=str, default='info', nargs='?', const='debug',
                            help='''set debug level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')
        parser.add_argument('--output-file', '-o', type=str, default='results.json', dest='output_filename',
                            help='''name of output file for recording JSON results (default=%(default)s)''')
        parser.add_argument('--choice-rand-seed', type=int, default=None, dest='choice_rand_seed',
                            help='''random seed for choices of subscribers & servers (default=%(default)s)''')
        parser.add_argument('--rand-seed', type=int, default=None, dest='rand_seed',
                            help='''random seed used by other classes via calls to random module (default=%(default)s)''')

        args = parser.parse_args(args)

        # convert to plain dict
        args = vars(args)
        failure_model = SmartCampusFailureModel(**args)
        args['failure_model'] = failure_model
        return cls(**args)

    def run_all_experiments(self):
        """Runs the requested experimental configuration
        for the requested number of times, saving the results to an output file."""

        # Log progress to a file so that we can check on
        # long-running simulations to see how far they've gotten.
        progress_filename = self.output_filename.replace(".json", ".progress")
        if progress_filename == self.output_filename:
            progress_filename += ".progress"
        try:
            progress_file = open(progress_filename, "w")
            progress_file.write("Starting experiments at time %s\n" % time.ctime())
        except IOError as e:
            log.warn("Error opening progress file for writing: %s" % e)
            progress_file = None

        # catch termination signal and immediately output results so we don't lose ALL that work
        def __sigint_handler(sig, frame):
            # HACK: try changing filename so we know it wasn't finished
            self.output_filename = self.output_filename.replace('.json', '_UNFINISHED.json')
            log.critical("SIGINT received! Outputting current results to %s and exiting" % self.output_filename)
            self.output_results()
            exit(1)
        signal.signal(signal.SIGINT, __sigint_handler)

        # start the actual experimentation
        for r in range(self.nruns):
            log.info("Starting run %d" % r)
            # QUESTION: should we really do this each iteration?  won't it make for higher variance?
            subs = self.choose_subscribers()
            pubs = self.choose_publishers()
            server = self.choose_server()
            failed_nodes, failed_links = self.apply_failure_model()
            # TODO: only need to do this each time for non-deterministic heuristics (or if we choose new pubs/subs each time)
            trees = self.build_mcast_trees(server, subs)
            result = self.run_experiment(failed_nodes, failed_links, server, pubs, subs, trees)
            result['run'] = r
            self.record_result(result)

            if progress_file is not None:
                try:
                    progress_file.write("Finished run %d at %s\n" % (r, time.ctime()))
                    progress_file.flush()  # so we can tail it
                except IOError as e:
                    log.warn("Error writing to progress file: %s" % e)
        self.output_results()

    def record_result(self, result):
        """Result is a dict that includes the percentage of subscribers
        reachable as well as metadata such as run #"""
        self.results['results'].append(result)

    def output_results(self):
        """Outputs the results to a file"""
        log.info("Results: %s" % json.dumps(self.results, sort_keys=True, indent=2))
        with open(self.output_filename, "w") as f:
            json.dump(self.results, f, sort_keys=True, indent=2)

    def apply_failure_model(self):
        """Removes edges/links from a copy of the topology according
        to the failure model.
        @:return modified networkx topology"""
        nodes, links = self.failure_model.apply_failure_model(self.topo)
        log.debug("Failed nodes: %s" % nodes)
        log.debug("Failed links: %s" % links)
        return nodes, links

    def choose_subscribers(self):
        hosts = self.topo.get_hosts()
        subs = self.random.sample(hosts, min(self.nsubscribers, len(hosts)))
        log.debug("Subscribers: %s" % subs)
        return subs

    def choose_publishers(self):
        hosts = self.topo.get_hosts()
        pubs = self.random.sample(hosts, min(self.npublishers, len(hosts)))
        log.debug("Publishers: %s" % pubs)
        return pubs

    def choose_server(self):
        server = self.random.choice(self.topo.get_servers())
        log.debug("Server: %s" % server)
        return server

    def build_mcast_trees(self, source, subscribers):
        """Build redundant multicast trees over the specified subscribers using
        the requested heuristic algorithm."""

        trees = self.topo.get_redundant_multicast_trees(source, subscribers, self.ntrees,
                                                        algorithm=self.mcast_heuristic[0],
                                                        heur_args=self.mcast_heuristic[1:])

        # Need to record which heuristic and tree # we used for later
        for tree in trees:
            tree.graph['heuristic'] = self.get_mcast_heuristic_name()

        # log.debug("MCast Trees: %s" % trees)

        return trees

    def get_mcast_heuristic_name(self):
        return self.build_mcast_heuristic_name(*self.mcast_heuristic)

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

    def choose_best_trees(self, failed_topology, server, publishers, subscribers, trees):
        """
        Picks the tree having the estimated best chance of reaching
        subscribers based on their overlap (or not) with the paths
        each publisher's sensor data packet arrived on.

        :param nx.Graph failed_topology:
        :param str server:
        :param List[str] publishers:
        :param List[str] subscribers:
        :param List[nx.Graph] trees: before failure
        :rtype dict:
        """

        # We'll return a dict of tree choices indexed by method used
        choices = dict()

        # Build up the Successfully Traversed Topology from each publisher
        # by determining which path the packet would take in the functioning
        # topology and add its edges to the STT only if that path is
        # functioning in the failed topology
        # NOTE: because we're using undirected graphs, we have to worry about
        # whether edge tuples are formatted (nodes ordered) properly, hence
        # we just add edges to the set object in both orders (u,v) and (v,u)
        stt = set()
        for pub in publishers:
            path = nx.shortest_path(self.topo.topo, pub, server, weight=DISTANCE_METRIC)
            if nx.is_simple_path(failed_topology, path):
                for u, v in zip(path, path[1:]):
                    stt.add((u,v))
                    stt.add((v,u))
        stt_graph = nx.Graph(e for e in stt)
        # NOTE: Need to ensure at least server is here as doing the is_simple_path
        # check below causes an error if it was never added (all failed).
        stt_graph.add_node(server)

        method = 'max-overlap'
        # IDEA: choose the tree with the most # edges overlapping the STT,
        # which means it has the most # 'known' working links.
        # We scale the total overlap by the number of edges in the tree
        # to avoid preferring larger trees that unnecessarily overlap
        # random paths that we don't care about.
        overlaps = [(len(stt.intersection(t.edges())) / float(nx.number_of_edges(t)),\
                     self.random.random(), t) for t in trees]
        choices[method] = max(overlaps)[2]

        method = 'min-missing'
        # IDEA: choose the tree with the lease # edges that haven't been
        # validated as 'currently functioning' by the publishers'
        # packets' paths, which lessens the probability that a link of
        # unknown status will have failed.
        # We use the size of a tree as a tie-breaker (prefer smaller ones)
        missing = [(len(set(t.edges()) - stt), nx.number_of_edges(t), self.random.random(), t) for t in trees]
        choices[method] = min(missing)[3]

        method = 'max-reachable'
        # IDEA: choose the tree with the most # reachable destinations,
        # as estimated by checking whether the path taken to each
        # destination is validated as 'currently functioning' by the STT
        dests_reachable = []
        for tree in trees:
            this_reachability = 0
            for sub in subscribers:
                path = nx.shortest_path(tree, server, sub, weight=DISTANCE_METRIC)
                if nx.is_simple_path(stt_graph, path):
                    this_reachability += 1
            dests_reachable.append((this_reachability, self.random.random(), tree))
        best = max(dests_reachable)
        choices[method] = best[2]

        method = 'importance'
        # IDEA: essentially a hybrid of max-overlap and max-reachable.
        # Instead of just counting # edges overlapping, count total
        # 'importance' of overlapping edges where the importance is
        # the # destination-paths traversing this edge.
        importance = []
        for tree in trees:
            # We'll use max-flow to find how many paths on each edge
            sink = "__choose_best_trees_sink_node__"
            for sub in subscribers:
                tree.add_edge(sub, sink, capacity=1)
            flow_value, flow = nx.maximum_flow(tree, server, sink)
            assert(flow_value == len(subscribers))  # else something wrong
            tree.remove_node(sink)

            # For every 'up' edge, count the flow along it as its importance.
            # Also divide by the total importance to avoid preferring larger trees
            this_importance = 0
            total_importance = 0.0
            for u, vd in flow.items():
                for v, f in vd.items():
                    if v == sink:
                        continue
                    if (u, v) in stt:
                        this_importance += f
                    total_importance += f
            importance.append((this_importance / total_importance, self.random.random(), tree))

        choices[method] = max(importance)[2]

        return choices

    def run_experiment(self, failed_nodes, failed_links, server, publishers, subscribers, trees):
        """Check what percentage of subscribers are still reachable
        from the server after the failure model has been applied
        by removing the failed_nodes and failed_links from each tree
        as well as a copy of the overall topology (for the purpose of
        establishing an upper bound oracle heuristic).

        We also explore the use of an intelligent multicast tree-choosing
        heuristic that picks the tree with the most overlap with the paths
        each publisher's sensor data packet arrived on.

        :param List[str] failed_nodes:
        :param List[str] failed_edges:
        :param str server:
        :param List[str] publishers:
        :param List[str] subscribers:
        :param List[nx.Graph] trees:
        :rtype dict:
        """

        # IDEA: we can determine the reachability by the following:
        # for each topology, remove the failed nodes and links,
        # determine all reachable nodes in topology,
        # consider only those that are subscribers,
        # record % reachable by any/all topologies.
        # We also have different methods of choosing which tree to use
        # and two non-multicast comparison heuristics.
        # NOTE: the reachability from the whole topology (oracle) gives us an
        # upper bound on how well the edge server could possibly do,
        # even without using multicast.

        subscribers = set(subscribers)
        result = dict()
        heuristic = self.get_mcast_heuristic_name()
        # we'll record reachability for various choices of trees
        result[heuristic] = dict()
        failed_topology = self.get_failed_topology(self.topo.topo, failed_nodes, failed_links)

        # ORACLE
        # First, use a copy of whole topology as the 'oracle' heuristic,
        # which sees what subscribers are even reachable by ANY path.
        failed_topology.graph['heuristic'] = 'oracle'
        topos_to_check = [failed_topology]
        reach = self.get_reachability(server, subscribers, topos_to_check)[0]
        result['oracle'] = reach

        # UNICAST
        # Second, get the reachability for the 'unicast' heuristic,
        # which sees what subscribers are reachable on the failed topology
        # via the path they'd normally be reached on the original topology
        paths = [nx.shortest_path(self.topo.topo, server, s, weight=DISTANCE_METRIC) for s in subscribers]
        # record the cost of the paths whether they would succeed or not
        unicast_cost = sum(self.topo.topo[u][v].get(COST_METRIC, 1) for p in paths\
                           for u, v in zip(p, p[1:]))
        # now filter only paths that are still functioning and record the reachability
        paths = [p for p in paths if nx.is_simple_path(failed_topology, p)]
        result['unicast'] = len(paths) / float(len(subscribers))

        # ALL TREES' REACHABILITIES: all, min, max, mean, stdev
        # Next, check all the redundant multicast trees together to get their respective (and aggregate) reachabilities
        topos_to_check = [self.get_failed_topology(t, failed_nodes, failed_links) for t in trees]
        reaches = self.get_reachability(server, subscribers, topos_to_check)
        heuristic = trees[0].graph['heuristic']  # we assume all trees from same heuristic
        result[heuristic]['all'] = reaches[-1]
        reaches = reaches[:-1]
        result[heuristic]['max'] = max(reaches)
        result[heuristic]['min'] = min(reaches)
        result[heuristic]['mean'] = np.mean(reaches)
        result[heuristic]['stdev'] = np.std(reaches)

        # CHOSEN
        # Finally, check the tree chosen by the edge server heuristic(s)
        # for having the best estimated chance of data delivery
        best_trees = self.choose_best_trees(failed_topology, server, publishers, subscribers, trees)
        for choice_method, best_tree in best_trees.items():
            best_tree_idx = trees.index(best_tree)
            reach = reaches[best_tree_idx]
            result[heuristic]['%s-chosen' % choice_method] = reach

        ### RECORDING METRICS ###
        # Record the distance to the subscribers in terms of # hops
        # TODO: make this latency instead?
        nhops = []
        for t in trees:
            for s in subscribers:
                nhops.append(len(nx.shortest_path(t, s, server)) - 1)
        result['nhops'] = dict(mean=np.mean(nhops), stdev=np.std(nhops), min=min(nhops), max=max(nhops))

        # Record the pair-wise overlap between the trees
        tree_edges = [set(t.edges()) for t in trees]
        overlap = [len(t1.intersection(t2)) for t1 in tree_edges for t2 in tree_edges]
        result['overlap'] = sum(overlap)

        # TODO: try to get this working on topos > 20?
        # the ILP will need some work if we're going to get even the relaxed version running on large topologies
        # overlap_lower_bound = ilp_redundant_multicast(self.topo.topo, server, subscribers, len(trees), get_lower_bound=True)
        # result['overlap_lower_bound'] = overlap_lower_bound

        # Record the average size of the trees
        costs = [sum(e[2].get(COST_METRIC, 1) for e in t.edges(data=True)) for t in trees]
        result['cost'] = dict(mean=np.mean(costs), stdev=np.std(costs), min=min(costs), max=max(costs),
                              unicast=unicast_cost)

        return result

    def get_reachability(self, server, subscribers, topologies):
        """Returns the average probability of reaching any of the subscribers from the
        server in each of the given topologies.  Also includes the result for using
        all topologies at once.
        :returns list: containing reachability for each topology (in order),
        with the last entry representing using all topologies at the same time
        """

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

    def get_failed_topology(self, topo, failed_nodes, failed_links):
        """Returns a copy of the graph topo with all of the failed nodes
        and links removed."""
        topology = topo.copy()
        topology.remove_edges_from(failed_links)
        topology.remove_nodes_from(failed_nodes)
        return topology

if __name__ == "__main__":
    import sys
    exp = SmartCampusNetworkxExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

