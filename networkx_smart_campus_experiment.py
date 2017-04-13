#! /usr/bin/python

# @author: Kyle Benson
# (c) Kyle Benson 2017

import json
import logging as log

import numpy as np
import networkx as nx

from smart_campus_experiment import SmartCampusExperiment
from ride.ride_d import RideD
from topology_manager.networkx_sdn_topology import NetworkxSdnTopology

COST_METRIC = 'weight'  # for links only
DISTANCE_METRIC = 'latency'  # for shortest path calculations
PUBLICATION_TOPIC = 'seismic_alert'


class NetworkxSmartCampusExperiment(SmartCampusExperiment):

    def __init__(self, *args, **kwargs):
        """
        :param float error_rate: error rate for PUBLICATIONS ONLY!
        :param args:
        :param kwargs:
        """
        super(NetworkxSmartCampusExperiment, self).__init__(*args, **kwargs)

    def record_result(self, result):
        """Result is a dict that includes the percentage of subscribers
        reachable as well as metadata such as run #"""
        self.results['results'].append(result)

    def output_results(self):
        """Outputs the results to a file"""
        log.info("Results: %s" % json.dumps(self.results, sort_keys=True, indent=2))
        with open(self.output_filename, "w") as f:
            json.dump(self.results, f, sort_keys=True, indent=2)

    def setup_topology(self):
        # only need to set this up once
        if self.topo is None:
            self.topo = NetworkxSdnTopology(self.topology_filename)
            # freeze graph to prevent any accidental topological changes
            nx.freeze(self.topo.topo)

    def run_experiment(self, failed_nodes, failed_links, server, publishers, subscribers):
        """Check what percentage of subscribers are still reachable
        from the server after the failure model has been applied
        by removing the failed_nodes and failed_links from each tree
        as well as a copy of the overall topology (for the purpose of
        establishing an upper bound oracle heuristic).

        We also explore the use of an intelligent multicast tree-choosing
        heuristic that picks the tree with the most overlap with the paths
        each publisher's sensor data packet arrived on.

        :param List[str] failed_nodes:
        :param List[str] failed_links:
        :param str server:
        :param List[str] publishers:
        :param List[str] subscribers:
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

        # start up and configure RideD middleware for building/choosing trees
        # We need to specify dummy addresses that won't actually be used for anything.
        addresses = ["10.0.0.%d" % d for d in range(self.ntrees)]
        rided = RideD(self.topo, server, addresses, self.ntrees, construction_algorithm=self.tree_construction_algorithm[0],
                      const_args=self.tree_construction_algorithm[1:])
        # HACK: since we never made an actual API for the controller, we just do this manually...
        for s in subscribers:
            rided.add_subscriber(s, PUBLICATION_TOPIC)
        # Build up the Successfully Traversed Topology (STT) from each publisher
        # by determining which path the packet would take in the functioning
        # topology and add its edges to the STT only if that path is
        # functioning in the failed topology.
        # BIG OH: O(T) + O(S), where S = |STT|
        for pub in publishers:
            path = nx.shortest_path(self.topo.topo, pub, server, weight=DISTANCE_METRIC)
            rided.set_publisher_route(pub, path)
            if self.random.random() >= self.error_rate and nx.is_simple_path(failed_topology, path):
                rided.notify_publication(pub)

        # build and get multicast trees
        trees = rided.build_mdmts()[PUBLICATION_TOPIC]
        # record which heuristic we used
        for tree in trees:
            tree.graph['heuristic'] = self.get_mcast_heuristic_name()
            # sanity check that the returned trees reach all destinations
            assert all(nx.has_path(tree, server, sub) for sub in subscribers)

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
        choices = dict()
        for method in RideD.TREE_CHOOSING_HEURISTICS:
            choices[method] = rided.get_best_mdmt(PUBLICATION_TOPIC, method)

        for choice_method, best_tree in choices.items():
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

    # The rest of the methods here are specific to this subclass (not inherited from base).

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

    @staticmethod
    def get_failed_topology(topo, failed_nodes, failed_links):
        """Returns a copy of the graph topo with all of the failed nodes
        and links removed."""
        # since we froze the graph we can't just use .copy()
        topology = nx.Graph(topo)
        topology.remove_edges_from(failed_links)
        topology.remove_nodes_from(failed_nodes)
        return topology

if __name__ == "__main__":
    import sys
    exp = NetworkxSmartCampusExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

