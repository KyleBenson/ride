# Resilient IoT Data Exchange - Dissemination middleware
import argparse

import time

import topology_manager

CLASS_DESCRIPTION =  """Middleware layer for sending reliable IoT event notifications to a group of subscribers.
    It configures and chooses from multiple Maximally-Disjoint Multicast Trees (MDMTs) based on
    knowledge of network state.  This state is based on recently received publications and the
    routes their packets took, which all comprise the Successfully Traversed Topology (STT)."""

import random

from stt_manager import SttManager
import networkx as nx
from topology_manager.sdn_topology import SdnTopology
import logging as log

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

    def __init__(self, topology_mgr, dpid, addresses, ntrees=2, tree_choosing_heuristic='importance',
                 tree_construction_algorithm=('red-blue',), **kwargs):
        """
        :param SdnTopology topology_mgr: used as adapter to SDN controller for
         maintaining topology and multicast tree information
        :param dpid: the data plane ID of the server this m/w runs on.  This MAY
        be some routable network address recognized by the SDN controller and MUST be included
        in the topology_manager.  It's the root of the multicast trees.  NOTE: a server
        with multiple interfaces will have to pick one of these (if using addresses for dpid),
        but you should configure your network in such a way that that the dpid can route to any
        of the interfaces along the best available path (e.g. add another IP address exposed on
        all interfaces, which we assume for a VM running the actual server).
        :param list[str] addresses: pool of IP addresses that we'll assign to MDMTs (NOTE: RIDE-D
        assumes that these addresses are routable by the networking stack!
          Use 'ip route [add]' to verify/configure).
        :param int ntrees:
        :param str tree_choosing_heuristic: which MDMT-choosing heuristic to use
        :param tuple[str] tree_construction_algorithm: which MDMT-construction algorithm to use (pos 0)
        followed by any args to that algorithm
        :param kwargs: ignored (just present so we can pass args from other classes without causing errors)
        """
        super(RideD, self).__init__()

        if len(addresses) != ntrees:
            raise ValueError("Must specify the same number of addresses as requested #multicast trees!")

        self.stt_mgr = SttManager()

        if not isinstance(topology_mgr, SdnTopology):
            # only adapter type specified: use default other args
            if isinstance(topology_mgr, basestring):
                self.topology_manager = topology_manager.build_topology_adapter(topology_adapter_type=topology_mgr)
            # we expect a dict to have the kwargs
            elif isinstance(topology_mgr, dict):
                self.topology_manager = topology_manager.build_topology_adapter(**topology_mgr)
            # hopefully it's a tuple!
            else:
                try:
                    self.topology_manager = topology_manager.build_topology_adapter(*topology_mgr)
                except TypeError:
                    raise TypeError("topology_mgr parameter is not of type SdnTopology and couldn't extract further parameters from it!")

        # ENHANCE: verify that it's a SdnTopology?  maybe accept a dict of args to its constructor?
        else:
            self.topology_manager = topology_mgr

        # ENHANCE: rather than manually specifying the DPID, we could iterate over the hosts in the
        # topology and find the one corresponding to our network stack.  However, the current method
        # is not only easier and less error-prone, but running part of RIDE-D on the SDN controller
        # would require directly specifying the server's ID (or having it provide this through some API) anyway.
        self.dpid = dpid
        # ENHANCE: who manages pool of IP addresses for MDMTs?  would need some controller API perhaps...
        self.address_pool = addresses
        self.ntrees = ntrees
        self.choosing_heuristic = tree_choosing_heuristic
        self.construction_algorithm = tree_construction_algorithm[0]
        self.const_args = tree_construction_algorithm[1:]

        # maps topic IDs to MDMTs, which are NetworkX graphs having an
        # attribute storing the address (IPv4?) of that tree
        self.mdmts = {}

        # maps publishers to the network routes their packets take to get here
        self.publisher_routes = {}

        # maps topic IDs to the subscribers
        self.subscribers = {}

    @classmethod
    def get_arg_parser(cls):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :return argparse.ArgumentParser arg_parser:
        """
        arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION,
                                             parents=(topology_manager.sdn_topology.SdnTopology.get_arg_parser(),),
                                             add_help=False)

        # Algorithmic configuration
        arg_parser.add_argument('--ntrees', '-t', type=int, default=2,
                                help='''number of redundant multicast trees to build (default=%(default)s)''')
        arg_parser.add_argument('--mcast-construction-algorithm', type=str, default=('steiner',), nargs='+',
                                dest='tree_construction_algorithm',
                                help='''heuristic algorithm for building multicast trees.  First arg is the heuristic
                                name; all others are passed as args to the heuristic. (default=%(default)s)''')
        arg_parser.add_argument('--choosing-heuristic', '-c', default='importance', dest='tree_choosing_heuristic',
                                help='''multicast tree choosing heuristic to use (default=%(default)s)''')

        # Networking-related configurations
        arg_parser.add_argument('--dpid', type=str, default='127.0.0.1',
                                help='''Data Plane ID (DPID) for the server, which is a unique identifier
                                (typically IP or MAC address) used by the SDN controller to identify
                                and address packets to the server on which RideD is running. (default=%(default)s)''')
        arg_parser.add_argument('--addresses', '-a', type=str, nargs='+', default=("127.0.0.1",),
                                help='''IP address(es) pool for the MDMTs. WARNING: RideD does not currently
                                support multiple topics fully!  Need to first add ability to use different
                                addresses and GroupIDs for different topics... default=%(default)s''')

        return arg_parser

    @classmethod
    def build_from_args(cls, args, pre_parsed=False):
        """Constructs from command line arguments.
        :param pre_parsed: if False, will first parse the given args using the ArgumentParser;
        if True, assumes args is the result of such a parsing
        """

        if not pre_parsed:
            args = cls.get_arg_parser().parse_args(args)

        topo_mgr = topology_manager.build_topology_adapter(args.topology_adapter_type, args.controller_ip, args.controller_port)
        args = vars(args) # convert to plain dict
        return cls(topology_manager=topo_mgr, **args)

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

    def notify_publication(self, publisher, at_time=None, id_type='dpid'):
        """
        Records that a publication successfully arrived at time at_time.

        :param str publisher: publisher identifier (e.g. IP Address)
        :param Datetime at_time: time the publication was received
        :param str id_type: one of ('ip', 'mac', 'dpid', 'id') that represents what type of identifier publisher is
         (NOTE: 'id' is an application-layer concept and 'dpid' is the ID that came from the SDN controller)
        :return:
        """
        log.debug("Received publication notification about publisher %s" % publisher)

        # First, convert publisher ID to a DPID
        if id_type == 'ip':
            publisher = self.topology_manager.get_host_by_ip(publisher)
        elif id_type == 'mac':
            publisher = self.topology_manager.get_host_by_mac(publisher)
        elif id_type == 'dpid':
            pass  # already correct
        elif id_type == 'id':
            raise NotImplementedError("Currently have no way of gathering application-layer publisher ID")
        else:
            raise ValueError("Unrecognized id_type: %s" % id_type)

        try:
            route = self.publisher_routes[publisher]
            return self.stt_mgr.route_update(route, at_time)
        except KeyError:
            # ignore as we just don't know about this publisher
            pass

    def build_mdmts(self):
        """Build redundant multicast trees over the specified subscribers using
        the requested heuristic algorithm."""

        source = self.get_server_id()
        for topic, subs in self.subscribers.items():
            # ENHANCE: include weight?
            trees = self.topology_manager.get_redundant_multicast_trees(
                source, subs, self.ntrees, algorithm=self.construction_algorithm, heur_args=self.const_args)

            for address, t in zip(self.address_pool, trees):
                self.set_address_for_mdmt(t, address)

            self.mdmts[topic] = trees

        return self.mdmts

    def install_mdmts(self, mdmts):
        """
        Installs the given MDMTs by pushing static flow rules to the SDN controller.
        WARNING: the IP addresses of the mdmts must be routable by the host!  Make sure
        you add them e.g. "ip route add 224.0.0.0/4 dev eth0"
        :param List[nx.Graph] mdmts:
        """

        # NOTE: We need to give the group a chance to be registered or else the flow rule will hang
        # as PENDING_ADD.  Thus, we install the groups immediately but then do the flows after all
        # groups were installed in order to give the controller time to commit them.
        flows = []

        for i,t in enumerate(mdmts):
            ip_address = self.get_address_for_mdmt(t)
            log.debug("Installing MDMT for %s" % ip_address)
            # TODO: need anything else here?  ip_proto=udp??? , ipv4_src=self.????
            matches = self.topology_manager.build_matches(ipv4_dst=ip_address)
            groups, flow_rules = self.topology_manager.build_flow_rules_from_multicast_tree(t, self.dpid, matches, group_id=i+10)
            for g in groups:
                # log.debug("Installing group: %s" % self.topology_manager.rest_api.pretty_format_parsed_response(g))
                self.topology_manager.install_group(g)
            flows.extend(flow_rules)

        time.sleep(2)
        for fr in flows:
            self.topology_manager.install_flow_rule(fr)

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

        # trees = self.build_mdmts()
        # self.install_mdmts(trees)
        # return
        # # TODO: maybe wrap this in a check to see if this is actually a live TopoMgr?  or just have a install multicast rules function that no-ops in an overridden version...

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
        """
        Return a list of all subscribers registered for the given topic.
        :param str topic_id:
        :raises KeyError: if the topic isn't registered
        :return List[str]:
        """
        return self.subscribers[topic_id]

    def set_publisher_route(self, publisher_id, route):
        """
        Adds the specified publisher host ID and its route within the local network
        to the list of currently publishing hosts,
        which is needed for calculating the STT.
        :param publisher_id: publisher's DPID
        :param list route: Each network node hop on the route including source and destination
        :return:
        """

        self.publisher_routes[publisher_id] = route

    # TODO: del_subscriber and del_publisher
    # ENHANCE: rather than having to manually pass the publisher/subscriber IP addresses
    # to RIDE-D, we could look them up through the topology_manager.  Of course, we'd still
    # need some ID for them...