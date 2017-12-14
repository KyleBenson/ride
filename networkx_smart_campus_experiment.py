#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2017

import logging
log = logging.getLogger(__name__)

import numpy as np
import networkx as nx

from smart_campus_experiment import SmartCampusExperiment, random, DISTANCE_METRIC
from ride.ride_d import RideD
from topology_manager.networkx_sdn_topology import NetworkxSdnTopology

COST_METRIC = 'weight'  # for links only
PUBLICATION_TOPIC = 'seismic_alert'


class NetworkxSmartCampusExperiment(SmartCampusExperiment):

    def __init__(self, *args, **kwargs):
        """
        :param float error_rate: error rate for PUBLICATIONS ONLY!
        :param args:
        :param kwargs:
        """
        super(NetworkxSmartCampusExperiment, self).__init__(*args, **kwargs)
        self.results['params']['experiment_type'] = 'networkx'

    def setup_topology(self):
        # only need to set this up once
        if self.topo is None:
            self.topo = NetworkxSdnTopology(self.topology_filename)
            # freeze graph to prevent any accidental topological changes
            nx.freeze(self.topo.topo)

    def run_experiment(self):
        """Check what percentage of subscribers are still reachable
        from the server after the failure model has been applied
        by removing the failed_nodes and failed_links from each tree
        as well as a copy of the overall topology (for the purpose of
        establishing an upper bound oracle heuristic).

        We also explore the use of an intelligent multicast tree-choosing
        heuristic that picks the tree with the most overlap with the paths
        each publisher's sensor data packet arrived on.

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

        subscribers = set(self.subscribers)
        result = dict()
        heuristic = self.get_mcast_heuristic_name()
        # we'll record reachability for various choices of trees
        result[heuristic] = dict()
        failed_topology = self.get_failed_topology(self.topo.topo, self.failed_nodes, self.failed_links)

        # start up and configure RideD middleware for building/choosing trees
        # We need to specify dummy addresses that won't actually be used for anything.
        addresses = ["10.0.0.%d" % d for d in range(self.ntrees)]
        rided = RideD(self.topo, self.server, addresses, self.ntrees, construction_algorithm=self.tree_construction_algorithm[0],
                      const_args=self.tree_construction_algorithm[1:])
        # HACK: since we never made an actual API for the controller, we just do this manually...
        for s in subscribers:
            rided.add_subscriber(s, PUBLICATION_TOPIC)

        # Build up the Successfully Traversed Topology (STT) from each publisher
        # by determining which path the packet would take in the functioning
        # topology and add its edges to the STT only if that path is
        # functioning in the failed topology.
        # BIG OH: O(T) + O(S), where S = |STT|

        # XXX: because the RideC implementation requires an actual SDN controller adapter, we just repeat the logic
        # for computing 'redirection' routes (publisher-->edge after cloud failure) here...
        if self.reroute_policy == 'shortest':
            pub_routes = {pub: self.topo.get_path(pub, self.server, weight=DISTANCE_METRIC) for pub in self.publishers}
        else:
            if self.reroute_policy != 'disjoint':
                log.error("unknown reroute_policy '%s'; defaulting to 'disjoint'...")
            pub_routes = {p[0]: p for p in self.topo.get_multi_source_disjoint_paths(self.publishers, self.server, weight=DISTANCE_METRIC)}
            assert list(sorted(pub_routes.keys())) == list(sorted(self.publishers)), "not all hosts accounted for in disjoint paths: %s" % pub_routes.values()

        for pub in self.publishers:
            path = pub_routes[pub]
            rided.set_publisher_route(pub, path)
            if random.random() >= self.error_rate and nx.is_simple_path(failed_topology, path):
                rided.notify_publication(pub)

        # build and get multicast trees
        trees = rided.build_mdmts()[PUBLICATION_TOPIC]
        # XXX: rather than use the install_mdmts API, which would try to install flow rules, we just set them directly
        rided.mdmts[PUBLICATION_TOPIC] = trees
        # record which heuristic we used
        for tree in trees:
            tree.graph['heuristic'] = self.get_mcast_heuristic_name()
            # sanity check that the returned trees reach all destinations
            assert all(nx.has_path(tree, self.server, sub) for sub in subscribers)

        # ORACLE
        # First, use a copy of whole topology as the 'oracle' heuristic,
        # which sees what subscribers are even reachable by ANY path.
        reach = self.get_oracle_reachability(subscribers, self.server, failed_topology)
        result['oracle'] = reach

        # UNICAST
        # Second, get the reachability for the 'unicast' heuristic,
        # which sees what subscribers are reachable on the failed topology
        # via the path they'd normally be reached on the original topology
        paths = [nx.shortest_path(self.topo.topo, self.server, s, weight=DISTANCE_METRIC) for s in subscribers]
        # record the cost of the paths whether they would succeed or not
        unicast_cost = sum(self.topo.topo[u][v].get(COST_METRIC, 1) for p in paths\
                           for u, v in zip(p, p[1:]))
        # now filter only paths that are still functioning and record the reachability
        paths = [p for p in paths if nx.is_simple_path(failed_topology, p)]
        result['unicast'] = len(paths) / float(len(subscribers))

        # TODO: disjoint unicast paths comparison!

        # ALL TREES' REACHABILITIES: all, min, max, mean, stdev
        # Next, check all the redundant multicast trees together to get their respective (and aggregate) reachabilities
        topos_to_check = [self.get_failed_topology(t, self.failed_nodes, self.failed_links) for t in trees]
        reaches = self.get_reachability(self.server, subscribers, topos_to_check)
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
        for method in RideD.MDMT_SELECTION_POLICIES:
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
                nhops.append(len(nx.shortest_path(t, s, self.server)) - 1)
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


if __name__ == "__main__":
    import sys
    exp = NetworkxSmartCampusExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

