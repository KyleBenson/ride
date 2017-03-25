# Resilient IoT Data Exchange - Dissemination middleware
import random

from stt_manager import SttManager
import networkx as nx
from topology_manager.sdn_topology import SdnTopology

class RideD(object):
    """
    Middleware layer for sending reliable IoT event notifications to a group of subscribers.
    It configures and chooses from multiple Maximally-Disjoint Multicast Trees (MDMTs) based on
    knowledge of network state.  This state is based on recently received publications and the
    routes their packets took, which all comprise the Successfully Traversed Topology (STT).

    RideC requires interaction with an SDN Controller to configure the trees and learn about
    topology information.

    RideC also requires interaction with the Data Exchange broker, which we assume is handled
    via some out-of-band mechanism (e.g. a REST API).

    NOTE: this version is designed as a Python class that runs in and is managed by the using
    applications main thread.  A more rigorous (commercialized) implementation would run as a
    middleware process on the application's machine and expose only a configuration API and
    the option to send a reliable notification.  It would instead call notify_publication() from
    a thread that intercepts publications as they arrive so as to obscure the complexity from
    the application.  It would also handle its own periodic update(), possibly based on
    subscribing to updates from the SDN controller.  Note also that the SDN Controller might
    implement a lot of the multicast tree construction/maintenance functionality as well as
    managing the list of publishers/subscribers for the Data Exchange broker.
    """

    TREE_CHOOSING_HEURISTICS = ('max-overlap', 'min-missing', 'max-reachable', 'importance')

    def __init__(self, topology_manager, dpid, ntrees=2, choosing_heuristic='importance',
                 construction_algorithm='red-blue', const_args=()):
        """
        :param SdnTopology topology_manager: used as adapter to SDN controller for
         maintaining topology and multicast tree information
        :param dpid: the data plane ID of the service this m/w runs on.  This will MAY
        be some routable network address recognized by the SDN controller and MUST be included
        in the topology_manager.  It's the root of the multicast trees.  NOTE: a server
        with multiple interfaces will have to pick one of these (if using addresses for dpid),
        but you should configure your network in such a way that that the dpid can route to any
        of the interfaces along the best available path (e.g. add another IP address exposed on
        all interfaces, which we assume for a VM running the actual server).
        :param int ntrees:
        :param str choosing_heuristic: which MDMT-choosing heuristic to use
        :param str construction_algorithm: which MDMT-construction algorithm to use
        :param tuple const_args: arguments to MDMT construction algorithm
        """
        super(RideD, self).__init__()
        self.stt_mgr = SttManager()

        self.topology_manager = topology_manager
        self.dpid = dpid
        self.ntrees = ntrees
        self.choosing_heuristic = choosing_heuristic
        self.construction_algorithm = construction_algorithm
        self.const_args = const_args

        # maps topic IDs to MDMTs, which are NetworkX graphs having an
        # attribute storing the address (IPv4?) of that tree
        self.mdmts = {}

        # maps publishers to the network routes their packets take to get here
        self.publisher_routes = {}

        # maps topic IDs to the subscribers
        self.subscribers = {}

    @staticmethod
    def get_address_for_mdmt(mdmt):
        return mdmt.graph['address']

    @staticmethod
    def set_address_for_mdmt(mdmt, address):
        mdmt.graph['address'] = address

    def get_server_id(self):
        """
        Returns the ID of the server for use with the topology manager.
        Currently just uses the DPID.
        """
        return self.dpid

    def get_best_multicast_address(self, topic, heuristic=None):
        return self.get_address_for_mdmt(self.get_best_mdmt(topic, heuristic))

    def get_best_mdmt(self, topic, heuristic=None):
        """
        The core of the RIDE-D middleware.  It determines which MDMT (multicast tree) to
        use for publishing to the specified topic and returns the network address used
        to send multicast packets along that tree.
        :param topic:
        :param str heuristic: identifies which heuristic method should be used (default=self.heuristic)
        :return network_address:
        """
        
        trees = self.mdmts[topic]
        subscribers = self.subscribers[topic]
        if heuristic is None:
            heuristic = self.choosing_heuristic
        root = self.get_server_id()

        # ENHANCE: could try using nx.intersection(G,H) but it requires the same nodes
        stt_set = self.stt_mgr.get_stt_edges()
        
        if heuristic == 'max-overlap':
            # IDEA: choose the tree with the most # edges overlapping the STT,
            # which means it has the most # 'known' working links.
            # We scale the total overlap by the number of edges in the tree
            # to avoid preferring larger trees that unnecessarily overlap
            # random paths that we don't care about.
            # BIG OH: O(k(T+S)) as we assume intersection done in O(|first| + |second|) time
            overlaps = [(len(stt_set.intersection(t.edges())) / float(nx.number_of_edges(t)),
                         random.random(), t) for t in trees]
            best = max(overlaps)[2]
            return best

        elif heuristic == 'min-missing':
            # IDEA: choose the tree with the lease # edges that haven't been
            # validated as 'currently functioning' by the publishers'
            # packets' paths, which lessens the probability that a link of
            # unknown status will have failed.
            # We use the size of a tree as a tie-breaker (prefer smaller ones)
            # BIG OH: O(k(T+S))
            missing = [(len(set(t.edges()) - stt_set), nx.number_of_edges(t), random.random(), t) for t in trees]
            best = min(missing)[3]
            return best

        elif heuristic == 'max-reachable':
            # IDEA: choose the tree with the most # reachable destinations,
            # as estimated by checking whether the path taken to each
            # destination is validated as 'currently functioning' by the STT
            # BIG OH (this implementation): O(S) + O(dijkstra(t)) + O(D) for each K = K(O(S+TlogT+T) + O(D)) = O(K(S+TlogT))
            #   -- if we do whole computation each time and were to make use of all-pairs paths,
            #      which this implementation does not.  Here's a better possible running time:
            # BIG OH (using intersect): O(K(T+S))
            #   -- take intersection of each T and STT: do a BFS on that for free (linear in size of intersection),
            #      outputting which subs reachable in that BFS starting at the root. The size of each tree's output
            #      is that tree's "reachability".

            # ENHANCE: should everyone use this?  or just put this up front?
            stt_graph = self.stt_mgr.get_stt()
            # NOTE: Need to ensure at least server is here as doing the is_simple_path
            # check below causes an error if it was never added (all failed).
            stt_graph.add_node(self.get_server_id())

            dests_reachable = []
            for tree in trees:
                this_reachability = 0
                for sub in subscribers:
                    # NOTE: only one path so no concern about length
                    path = nx.shortest_path(tree, root, sub)
                    if nx.is_simple_path(stt_graph, path):
                        this_reachability += 1
                dests_reachable.append((this_reachability, random.random(), tree))
            best = max(dests_reachable)[2]
            return best

        elif heuristic == 'importance':
            # IDEA: essentially a hybrid of max-overlap and max-reachable.
            # Instead of just counting # edges overlapping, count total
            # 'importance' of overlapping edges where the importance is
            # the # destination-paths traversing this edge.
            # BIG-OH for a revised intersection-based version:
            # O(K(T+S)) by basically pre-computing the 'importance' of each tree edge
            #      via BFS/DFS where we count #children for each node
            importance = []
            for tree in trees:
                # We'll use max-flow to find how many paths on each edge
                sink = "__choose_best_trees_sink_node__"
                for sub in subscribers:
                    tree.add_edge(sub, sink, capacity=1)
                flow_value, flow = nx.maximum_flow(tree, root, sink)
                assert (flow_value == len(subscribers))  # else something wrong
                tree.remove_node(sink)

                # For every 'up' edge, count the flow along it as its importance.
                # Also divide by the total importance to avoid preferring larger trees
                this_importance = 0
                total_importance = 0.0
                for u, vd in flow.items():
                    for v, f in vd.items():
                        if v == sink:
                            continue
                        if (u, v) in stt_set:
                            this_importance += f
                        total_importance += f
                importance.append((this_importance / total_importance, random.random(), tree))

            best = max(importance)[2]
            return best

        else:
            raise ValueError("Unrecognized heuristic method type requested: %s" % heuristic)

    def notify_publication(self, publisher, at_time=None):
        """
        Records that a publication successfully arrived at time at_time.

        :param str publisher: publisher identifier (e.g. IP Address)
        :param Datetime at_time: time the publication was received
        :return:
        """

        route = self.publisher_routes[publisher]
        return self.stt_mgr.route_update(route, at_time)

    def build_mdmts(self):
        """Build redundant multicast trees over the specified subscribers using
        the requested heuristic algorithm."""

        source = self.get_server_id()
        for topic, subs in self.subscribers.items():
            # ENHANCE: include weight?
            trees = self.topology_manager.get_redundant_multicast_trees(
                source, subs, self.ntrees, algorithm=self.construction_algorithm, heur_args=self.const_args)
            self.mdmts[topic] = trees

        return self.mdmts

    def update(self):
        """
        Tells RideD to update itself by getting the latest subscribers, publishers,
        publication routes, and topology.  It rebuilds and reinstalls the multicast
        trees if necessary.
        :return:
        """
        raise NotImplementedError

        # NO UPDATES TO ACTUALLY GATHER HERE:
        # TODO: since we currently assume static pubs/subs/routes/topology, need to handle them dynamically
        # This requires some interaction with the controller/data exchange beyond that provided by
        # a regular controller and its REST APIs: e.g. who manages pool of IP addresses for MDMTs?
        # We'd also need to extend the REST APIs to support updating the topology rather than getting a whole new one.
        # We'd also have to handle dynamic pub/sub join/leave in this class

        # self.build_mdmts()
        # return
        # # TODO: assign ip addresses
        # # TODO: maybe wrap the below in a check to see if this is actually a live TopoMgr?  or just have a install multicast rules function that no-ops...
        # # install flow rules for each tree
        # for t in trees:
        #     matches = ????
        #     flow_rules = self.topology_manager.get_flow_rules_from_multicast_tree(t, source, matches)  # group_id????
        #     for fr in flow_rules:
        #         # TODO: ensure we don't need to separately install group_rules?  if so we need a refactor of the SdnTopology...
        #         # ENHANCE: handle errors
        #         self.topology_manager.install_flow_rule(fr)

    def add_subscriber(self, subscriber, topic_id):
        """
        Adds the specified subscriber host ID to the list of hosts currently subscribed
        to the given topic.  This is needed for calculating the multicast trees.
        :param subscriber:
        :param topic_id:
        :return:
        """

        self.subscribers.setdefault(topic_id, []).append(subscriber)

    def get_subscribers_for_topic(self, topic_id):
        return self.subscribers[topic_id]

    def set_publisher_route(self, publisher_id, route):
        """
        Adds the specified subscriber host ID to the list of currently subscribed hosts,
        which is needed for calculating the multicast trees.
        :param publisher_id:
        :param list route: Each network node hop on the route including source and destination
        :return:
        """

        self.publisher_routes[publisher_id] = route

    # TODO: del_subscriber and del_publisher