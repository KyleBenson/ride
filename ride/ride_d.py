# Resilient IoT Data Exchange - Dissemination middleware

CLASS_DESCRIPTION =  """Middleware layer for sending reliable IoT event notifications to a group of subscribers.
    It configures and chooses from multiple Maximally-Disjoint Multicast Trees (MDMTs) based on
    knowledge of network state.  This state is based on recently received publications and the
    routes their packets took, which all comprise the Successfully Traversed Topology (STT)."""

import argparse
import time
from threading import Lock, Thread, ThreadError
import random

import networkx as nx

import topology_manager
from ride.config import MULTICAST_FLOW_RULE_PRIORITY
from stt_manager import SttManager
from topology_manager.sdn_topology import SdnTopology

import logging
log = logging.getLogger(__name__)

class RideD(object):
    """
    Middleware layer for sending reliable IoT event notifications to a group of subscribers.
    It configures and chooses from multiple Maximally-Disjoint Multicast Trees (MDMTs) based on
    knowledge of network state.  This state is based on recently received publications and the
    routes their packets took, which all comprise the Successfully Traversed Topology (STT).

    An alert is sent via some underlying networking mechanism by calling a user-specified __send_to function
    (i.e. likely a UDP network socket for multicasting).  Its life-cycle is managed as an AlertContext object...

    RideD requires interaction with an SDN Controller to configure the trees and learn about
    topology information.

    RideD also requires interaction with the Data Exchange broker, which we assume is handled
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

    # Names of the MDMT selection policies
    MAX_LINK_IMPORTANCE = 'importance'
    MAX_REACHABLE_SUBSCRIBERS = 'max-reachable'
    MIN_MISSING_LINKS = 'min-missing'
    MAX_OVERLAPPING_LINKS = 'max-overlap'
    MDMT_SELECTION_POLICIES = (MAX_OVERLAPPING_LINKS, MIN_MISSING_LINKS, MAX_REACHABLE_SUBSCRIBERS, MAX_LINK_IMPORTANCE)

    def __init__(self, topology_mgr, dpid, addresses, ntrees=2, tree_choosing_heuristic=MAX_LINK_IMPORTANCE,
                 tree_construction_algorithm=('red-blue',), alert_sending_callback=None, max_retries=None, **kwargs):
        """
        :param SdnTopology|str topology_mgr: used as adapter to SDN controller for
         maintaining topology and multicast tree information
        :param dpid: the data plane ID of the server this m/w runs on.  This MAY
        be some routable network address recognized by the SDN controller and MUST be included
        in the topology_manager.  It's the root of the multicast trees.  NOTE: a server
        with multiple interfaces will have to pick one of these (if using addresses for dpid),
        but you should configure your network in such a way that that the dpid can route to any
        of the interfaces along the best available path (e.g. add another IP address exposed on
        all interfaces, which we assume for a VM running the actual server).
        :param list[str] addresses: pool of network addresses (e.g. (ipv4, udp_src_port#) tuples) that we'll assign to MDMTs
                (NOTE: RIDE-D assumes that these addresses are routable by the networking stack!
                Use 'ip route [add]' to verify/configure).
                ****NOTE: we use udp_src_port rather than the expected dst_port because this allows the clients to
                 respond to this port# and have the response routed via the proper MDMT
        :param int ntrees:
        :param str tree_choosing_heuristic: which MDMT-choosing heuristic to use
        :param tuple[str] tree_construction_algorithm: which MDMT-construction algorithm to use (pos 0)
        followed by any args to that algorithm
        :param alert_sending_callback: if specified, will be used as the callback to actually send the raw alert data
            through the underlying networking channel via the best current MDMT (likely its network address: IPv4/UDP port #);
            if unspecified, you MUST subclass this class and implement this method as:
            "__try_send_alert_packet_via(self, alert_context, mdmt)" to properly enable sending alerts.
            NOTE: you should handle setting up callbacks for subscriber responses that notify RideD of a successful
            delivery in this callback!  Consider just attaching it to the AlertContext
            NOTE: make sure you make this callback thread-safe if it's asynchronous!  The AlertContext.thread_lock will
            be locked when it's called so be careful accessing it or you might deadlock!
        :param max_retries: number of times sending an alert will be retried (using a different MDMT each time).
            default=2*ntrees
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

        # manages the AlertContext objects currently outstanding
        self._alerts = set()

        self.__try_send_alert_packet_via = alert_sending_callback
        self.max_retries = max_retries if max_retries is not None else 2 * ntrees

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
        arg_parser.add_argument('--choosing-heuristic', '-c', default=cls.MAX_LINK_IMPORTANCE, dest='tree_choosing_heuristic',
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
        # Also set the name so we can easily and uniquely identify the MDMT
        mdmt.name = address

    def get_server_id(self):
        """
        Returns the ID of the server for use with the topology manager.
        Currently just uses the DPID.
        """
        return self.dpid

    def send_alert(self, msg, topic, **retransmit_kwargs):
        """
        Resiliently send the specified message to the given alert topic by creating an AlertContext and using it to
        choose the best available MDMT and continually re-send packets over diverse paths until they arrive at all
        subscribers.
        :param msg:
        :param topic:
        :param retransmit_kwargs: keyword arguments sent to _alert_retransmit_loop(...); to disable retransmission
            specify max_retries=0
        :return: the alert being sent
        :rtype: RideD.AlertContext
        """

        alert_ctx = self._make_new_alert(msg, topic)
        self._do_send_alert(alert_ctx)

        # start the retransmit_loop, but we don't need to save the thread since we won't directly join it
        # but rather just let it expire after calling cancel_alert()
        thread = Thread(target=self._alert_retransmit_loop, args=(alert_ctx,), kwargs=retransmit_kwargs,
                        name="retx_thread_%s" % alert_ctx)
        thread.start()

        return alert_ctx

    def _do_send_alert(self, alert_ctx):
        """
        Chooses the best method (available MDMTs or just unicast) for sending this alert and then actually send the
        packet using the underlying network mechanism.
        :param alert_ctx:
        :return: the MDMT chosen for sending the alert
        """

        mdmt = self.get_best_mdmt(alert_ctx)
        alert_ctx.record_mdmt_used(mdmt)
        log.debug("sending alert via best available MDMT: %s" % str(mdmt.name))

        # we defer to this implementation-specific message-sending function to ease the integration of various
        # networks/protocols for actually sending the alert messages on the wire
        self.__try_send_alert_packet_via(alert_ctx, mdmt)

        return mdmt

    def _alert_retransmit_loop(self, alert_ctx, timeout=2, max_retries=None):
        """
        Repeatedly tries to send the specified alert every timeout seconds until either attempting it max_retries times,
        successfully delivering the alert to all subscribers, or cancel_alert() is explicitly called.
        :param alert_ctx:
        :type alert_ctx: RideD.AlertContext
        :param timeout: timeout between attempts in seconds
        :param max_retries: default=self.max_retries(which by default is 2*(#MDMTs))
        :return:
        """

        if max_retries == 0:
            return
        elif max_retries is None:
            max_retries = self.max_retries

        retry_attempts = 0
        while alert_ctx.active and max_retries > retry_attempts:
            log.debug("waiting %.2f secs to retransmit alert %s" % (timeout, alert_ctx))
            # Sleep first since we've already attempted to send the alert and each attempt here is a RE-try
            time.sleep(timeout)

            # We need to acquire the lock and then check again if the alert is still active or else we might re-try
            # after it's been canceled but it was blocked by the thread lock or sleep statement (it seems still active).
            # WARNING: this will lock out anything from getting called that needs the thread_lock, so if we
            # update the API to use the lock when e.g. record_mdmt_used then we'll need to add more locks...
            with alert_ctx.thread_lock:
                if alert_ctx.active:
                    log.debug("retransmitting alert %s (attempt #%d)" % (alert_ctx, retry_attempts))
                    self._do_send_alert(alert_ctx)
                    retry_attempts += 1

        # this means that the alert wasn't finished or cancelled, so we must've hit max_retries!
        if alert_ctx.active:
            # NOTE: make sure to sleep and give this last attempt a chance to reach the subscribers!
            time.sleep(timeout)
            # WARNING: As above, should probably acquire the lock before checking its status but that would deadlock
            # once we call cancel_alert(), so we should probably just tolerate multiple cancel calls...
            if alert_ctx.active:
                log.info("alert %s expired after %d re-tries..." % (alert_ctx, max_retries))
                self.cancel_alert(alert_ctx, success=False)

    def notify_alert_response(self, responder, alert_ctx, mdmt_used):
        """
        Updates view of current network topology state based on the route traveled by this response.  Also records that
        this subscriber was reached so that future re-tries don't worry about reaching it.
        :param responder:
        :param alert_ctx:
        :type alert_ctx: RideD.AlertContext
        :param mdmt_used: we require this instead of gleaning it from the alert_ctx since this response may have been
        sent before the most recent alert attempt (i.e. we might calculate the response route wrong)
        :return:
        """

        # determine the path used by this response and notify RideD that it is currently functional
        route = nx.shortest_path(mdmt_used, self.get_server_id(), responder)
        log.debug("processing alert response via route: %s" % route)

        # NOTE: this likely won't do much as we probably already selected this MDMT since this route was functional...
        self.stt_mgr.route_update(route)

        alert_ctx.record_subscriber_reached(responder)

        if not alert_ctx.has_unreached_subscribers():
            log.info("alert %s successfully reached all subscribers! closing..." % alert_ctx)
            self.cancel_alert(alert_ctx, success=True)

    def get_best_multicast_address(self, alert_context, heuristic=None):
        return self.get_address_for_mdmt(self.get_best_mdmt(alert_context, heuristic))

    def get_best_mdmt(self, alert_context, heuristic=None):
        """
        The core of the RIDE-D middleware.  It determines which MDMT (multicast tree) to
        use for publishing to the specified topic and returns the network address used
        to send multicast packets along that tree.
        :param alert_context: used to determine which subscribers, MDMTs, etc. are available and should be chosen
        :type alert_context: RideD.AlertContext
        :param heuristic: identifies which heuristic method should be used (default=self.choosing_heuristic)
        :type heuristic: str
        :return network_address:
        """

        if heuristic is None:
            heuristic = self.choosing_heuristic

        # We only consider the unreached subscribers so as to potentially weight MDMTs with unexplored diverse paths
        # better suited to reaching only the unreached subscribers.
        subscribers = alert_context.unreached_subscribers()

        root = self.get_server_id()
        mdmts = alert_context.mdmts

        # To only consider branches of the MDMTs used for unreached subscribers, we need to trim them down.
        # IDEA: we compute 'importance' with only a subset of the subscribers (unreached ones), trim off any edges
        # with 0 importance, and use the resulting tree as both the MDMTs and also the importance graph
        if len(subscribers) < len(alert_context.subscribers):
            trees = []
            for tree in mdmts:
                tree = self.get_importance_graph(tree, subscribers, root)

                tree.remove_edges_from([(u, v) for u, v, imp in tree.edges(data=self.IMPORTANCE_ATTRIBUTE_NAME) if (imp == 0)])
                # NOTE: because we only consider edges in these metrics, we don't need to cut out the nodes too
                trees.append(tree)

        # None reached yet, so no need to trim...
        # BUT, ensure we've calculated the importance if that's the metric we're using!
        elif heuristic == self.MAX_LINK_IMPORTANCE:
            trees = [self.get_importance_graph(tree, subscribers, root) for tree in mdmts]
        else:
            trees = mdmts

        # ENHANCE: could try using nx.intersection(G,H) but it requires the same nodes
        stt_set = self.stt_mgr.get_stt_edges()
        
        if heuristic == self.MAX_OVERLAPPING_LINKS:
            # IDEA: choose the tree with the most # edges overlapping the STT,
            # which means it has the most # 'known' working links.
            # We scale the total overlap by the number of edges in the tree
            # to avoid preferring larger trees that unnecessarily overlap
            # random paths that we don't care about.
            # BIG OH: O(k(T+S)) as we assume intersection done in O(|first| + |second|) time
            metrics = [(len(stt_set.intersection(t.edges())) / float(nx.number_of_edges(t)),
                        random.random(), t) for t in trees]
            mdmt_index = 2

        elif heuristic == self.MIN_MISSING_LINKS:
            # IDEA: choose the tree with the lease # edges that haven't been
            # validated as 'currently functioning' by the publishers'
            # packets' paths, which lessens the probability that a link of
            # unknown status will have failed.
            # We use the size of a tree as a tie-breaker (prefer smaller ones)
            # BIG OH: O(k(T+S))
            # NOTE: we use negative numbers here so that we can just apply a max function later as for the other metrics
            metrics = [(-len(set(t.edges()) - stt_set), nx.number_of_edges(t), random.random(), t) for t in trees]
            mdmt_index = 3

        elif heuristic == self.MAX_REACHABLE_SUBSCRIBERS:
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

            metrics = []
            for tree in trees:
                this_reachability = 0
                for sub in subscribers:
                    # NOTE: only one path so no concern about length
                    path = nx.shortest_path(tree, root, sub)
                    if nx.is_simple_path(stt_graph, path):
                        this_reachability += 1
                metrics.append((this_reachability, random.random(), tree))
            mdmt_index = 2

        elif heuristic == self.MAX_LINK_IMPORTANCE:
            # IDEA: essentially a hybrid of max-overlap and max-reachable.
            # Instead of just counting # edges overlapping, count total
            # 'importance' of overlapping edges where the importance is
            # the # destination-paths traversing this edge.
            # Also divide by the total importance to avoid preferring larger trees
            # BIG-OH (using intersection):
            # O(K(T+S)) by basically pre-computing the 'importance' of each tree edge
            #      via BFS/DFS where we count #children for each node
            # NOTE: we already computed the 'importance' in the beginning of this method!
            metrics = []

            for tree in trees:
                up_edges = stt_set.intersection(tree.edges())
                this_importance = sum((tree[u][v][self.IMPORTANCE_ATTRIBUTE_NAME] for u, v in up_edges))
                total_importance = sum((imp for u, v, imp in tree.edges(data=self.IMPORTANCE_ATTRIBUTE_NAME)))
                final_importance = float(this_importance) / float(total_importance) if total_importance != 0 else 0
                metrics.append((final_importance, random.random(), tree))

            mdmt_index = 2

        else:
            raise ValueError("Unrecognized heuristic method type requested: %s" % heuristic)

        # Now we can choose which the best MDMT was based on the metrics.  We do this here because we don't want to
        # repeat the logic that checks if this MDMT was selected recently and should be skipped over for now despite
        # having the best perceived metric value.  This approach ensures re-trying with different MDMTs despite little
        # or no new information about network state.
        metrics = sorted(metrics, reverse=True)

        # Work our way from best candidate to worst and select the first that we haven't used recently.
        # Since we make a copy of the MDMTs each time, they won't be the exact same graph between calls to this method.
        # Hence, we need to compare them by name to keep their ID consistent.
        recent_mdmts_used = {t.name for t in alert_context.most_recently_used_mdmts()}
        for candidate in metrics:
            tree = candidate[mdmt_index]
            if tree.name not in recent_mdmts_used:
                best = tree
                break
        else:
            raise RuntimeError("why did we never select one of the MDMTs for use?  Something's wrong here...")

        log.debug("selected MDMT '%s' via policy '%s' with metric value: %f" % (best.name, heuristic, candidate[0]))

        # We need to return the ORIGINAL MDMT rather than the trimmed down one
        for t in mdmts:
            if t.name == best.name:
                best = t
                break
        else:
            raise RuntimeError("selected best MDMT with name %s not found in original MDMTs %s" % (best.name, str(mdmts)))

        return best

    # for identifying the attribute in the importance graphs that stores the 'link-importance' metric
    IMPORTANCE_ATTRIBUTE_NAME = 'ride_d_link_importance'

    @classmethod
    def get_importance_graph(cls, tree, subscribers, root, make_copy=True):
        """
        Computes the 'link-importance' metric for the given multicast tree, subscribers, and root.  Stores this metric
        in the edges' attributes.  The 'link-importance' is basically the number of root-to-subscriber paths that pass
        through this edge; it represents the resilience importance of this link.

        This method basically uses a modified depth-first search to calculate the importance of each link after it
        visits the nodes i.e. when considering 'reverse edges'.

        :param tree:
        :type tree: nx.Graph
        :param subscribers: collection of subscribers (ideally a set for fast presence querying)
        :param root:
        :param make_copy: if True, returns a copy of tree rather than storing the 'importance' directly as an attribute
        :return: the tree with every edge having a link importance attribute
        """

        # This ensures the original graph doesn't have an importance attribute leaked between executions of this method
        if make_copy:
            tree = tree.copy()

        # we'll track the outgoing edges from each node so that we can sum up their importances
        outgoing_edges = dict()

        # Do the modified DFS and calculate importance on way 'back up tree'
        for u, v, edge_type in nx.dfs_labeled_edges(tree, source=root):
            if u == v or edge_type == 'nontree':
                continue
            elif edge_type == 'forward':
                outgoing_edges.setdefault(u, []).append((u, v))
            else:
                # We know it's a reverse edge now so we're calculating importance.
                # On the way back 'up the tree', we know there's only one incoming edge so when we consider an edge as
                # incoming, we can easily sum up the other outgoing edges as they've been assigned importance already.
                # We'll also increment the importance by 1 when we hit a subscriber.  Note that this should handle non-leaf
                # subscribers, but we don't really consider that case so it isn't fully tested...
                imp = sum((tree[_u][_v][cls.IMPORTANCE_ATTRIBUTE_NAME] for _u, _v in outgoing_edges.get(v, [])),
                          1 if v in subscribers else 0)
                tree[u][v][cls.IMPORTANCE_ATTRIBUTE_NAME] = imp

        # Verify all edges have some importance as we promised...
        if __debug__:
            # Also verify it's actually a tree as otherwise that'd explain edges without an importance...
            if not nx.is_tree(tree):
                raise RuntimeError("MDMT %s is not a tree!  Edges: %s" % (tree.name, list(tree.edges())))
            edges_no_importance = [(u,v) for u, v, imp in tree.edges(data=cls.IMPORTANCE_ATTRIBUTE_NAME) if imp is None]
            if edges_no_importance:
                raise RuntimeError("MDMT %s has edges with no importance: %s" % (tree.name, edges_no_importance))

        return tree

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
            log.debug("updating STT with functional route: %s" % route)
            return self.stt_mgr.route_update(route, at_time)
        except KeyError:
            # ignore as we just don't know about this publisher
            log.debug("publisher %s not found!  skipping... options are: %s" % (publisher, self.publisher_routes))
            pass

    def build_mdmts(self, subscribers=None):
        """
        Build redundant multicast trees over the specified subscribers (and relevant topics) using the configured heuristic algorithm.
        :param subscribers: dict mapping topics to a list of subscriber addresses for that topic (default=self.subscribers)
        :type subscribers: dict[str, List[str]]
        :return: an updated dict mapping topics to a list of MDMTs for reaching that topic's subscribers
        :rtype: dict[str, List[nx.Graph]]
        """

        if subscribers is None:
            subscribers = self.subscribers

        source = self.get_server_id()
        mdmts = dict()

        for topic, subs in subscribers.items():
            # XXX: ensure all subscribers are present in the topology to prevent e.g. KeyErrors from the various algorithms
            subs = [s for s in subs if s in self.topology_manager.topo]
            # TODO: check for reachability?  log error if they aren't available?

            # ENHANCE: include weight?
            trees = self.topology_manager.get_redundant_multicast_trees(
                source, subs, self.ntrees, algorithm=self.construction_algorithm, heur_args=self.const_args)

            mdmts[topic] = trees

        return mdmts

    def install_mdmts(self, mdmts, address_pool=None):
        """
        Installs the given MDMTs by pushing static flow rules to the SDN controller.
        Also sets the IP addresses of the MDMTs from those specified (or self.address_pool if unspecified)
        WARNING: the IP addresses of the mdmts must be routable by the host!  Make sure
        you add them e.g. "ip route add 224.0.0.0/4 dev eth0"
        :param List[nx.Graph] mdmts:
        :param List[str] address_pool: list of network addresses from which to assign the MDMTs their addresses.  Note
        that they must have the same length!  default=self.address_pool
        """

        if address_pool is None:
            address_pool = self.address_pool

        # NOTE: We need to give the group a chance to be registered or else the flow rule will hang
        # as PENDING_ADD.  Thus, we install the groups immediately but then do the flows after all
        # groups were installed in order to give the controller time to commit them.
        flows = []

        # ENHANCE: perhaps we need to remove old flows before installing these ones? in our current setting we just overwrite them for the most part...
        # ENHANCE: not re-install ones that are the same? would make a small performance improvement

        if len(address_pool) < len(mdmts):
            log.warning("requested to install %d MDMTs but only provided %d network addresses to assign them!"
                        " Will install as many as we have addresses..." % (len(address_pool), len(mdmts)))

        for i, t, address in zip(range(len(mdmts)), mdmts, address_pool):
            self.set_address_for_mdmt(t, address)
            log.debug("Installing MDMT for address %s" % str(address))
            matches = self.build_flow_matches_from_address(address)
            # XXX: we need to include the UDP port so that hosts' responses can be routed via different MDMTs
            response_matching = {"udp_dst": address[1]}
            groups, flow_rules = self.topology_manager.build_flow_rules_from_multicast_tree(t, self.dpid, matches,
                                                                                            group_id=i+10,
                                                                                            priority=MULTICAST_FLOW_RULE_PRIORITY,
                                                                                            route_responses=response_matching)
            for g in groups:
                # log.debug("Installing group: %s" % self.topology_manager.rest_api.pretty_format_parsed_response(g))
                res = self.topology_manager.install_group(g)
                if not res:
                    log.error("Problem installing group %s" % g)
            flows.extend(flow_rules)

        # Need a chance for groups to populate or the flow rule will have an unknown group treatment!
        # ENHANCE: could wait for the groups to actually finish populating?
        time.sleep(2)
        if not self.topology_manager.install_flow_rules(flows):
            log.error("Problem installing flow rules: %s" % flows)

    def build_flow_matches_from_address(self, address):
        """
        Builds flow matching objects from the given address, which by default is assumed to be a tuple of:
         (ipv4_addr, udp_src_port_num)  --  Override this method to use different addresses
        :param address:
        :return:
        """
        # TODO: need anything else here?  ip_proto=udp???  udp_dst???
        assert isinstance(self.topology_manager, SdnTopology)
        src_ip = self.topology_manager.get_ip_address(self.get_server_id())
        matches = self.topology_manager.build_matches(ipv4_src=src_ip, ipv4_dst=address[0], udp_src=address[1])
        return matches

    def update(self):
        """
        Tells RideD to update itself by getting the latest subscribers, publishers,
        publication routes, and topology.  It rebuilds and reinstalls the multicast
        trees if necessary.
        :return:
        """

        # ENHANCE: extend the REST APIs to support updating the topology rather than getting a whole new one.
        self.topology_manager.build_topology(from_scratch=True)

        # TODO: need to invalidate outstanding alerts if the MDMTs change!  or at least invalidate their changed MDMTs...

        # XXX: during lots of failures, the updated topology won't see a lot of the nodes so we'll be catching errors...
        trees = None
        try:
            trees = self.build_mdmts()
            # TODO: maybe we should only save the built MDMTs as we add their flow rules? this could ensure that any MDMT we try to use will at least be fully-installed...
            # could even use a thread lock to block until the first one is installed
            self.mdmts = trees
        except nx.NetworkXError as e:
            log.error("failed to create MDMTs (likely due to topology disconnect) due to error: \n%s" % e)

        if trees:
            # ENHANCE: error checking/handling esp. for the multicast address pool that must be shared across all topics!
            for mdmts in trees.values():
                try:
                    self.install_mdmts(mdmts)
                except nx.NetworkXError as e:
                    log.error("failed to install_mdmts due to error: %s" % e)
        elif self.subscribers:
            log.error("empty return value from build_mdmts() when we do have subscribers!")
        # ENHANCE: retrieve publication routes rather than rely on them being manually set...

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


    ################################################################################################
    ########################   ALERT CONTEXT OBJECT MANAGEMENT     #################################
    ################################################################################################

    alert_id = 0

    def _make_new_alert(self, msg, topic):
        """
        Create a new alert context for the given topic that will manage the reliable delivery of the specified message.
        :param msg:
        :param topic:
        :return: the new alert
        :rtype: RideD.AlertContext
        """

        # ENHANCE: thread-safe locking so multiple alerts can be sent from different threads simultaneously

        subs = self.get_subscribers_for_topic(topic)
        mdmts = self.mdmts[topic]
        alert = self.AlertContext(msg, topic, subs, mdmts, self.alert_id)
        self.alert_id += 1
        self._alerts.add(alert)
        return alert

    def cancel_alert(self, alert, success=False):
        """
        Gracefully cancels the specified alert, frees any associated resources when possible/necessary, and stops
         retransmissions.  Note that if a retransmission attempt is currently in progress it will finish before cancel is run.
        Also note that multiple calls to cancel_alert is tolerable, though the thread_lock will make them run synchronously.
        :param alert:
        :param success: True if all of the subscribers were successfully reached (currently un-used)
        :type alert: RideD.AlertContext
        """

        log.debug("cancelling alert %s that was finished %s" % (alert, "successfully" if success else "unsuccessfully"))

        # First, we need to acquire the lock to ensure we aren't trying to cancel it in the middle of a retransmission
        with alert.thread_lock:
            # Stop the retransmission loop
            alert.active = False

            # QUESTION: any other resources to clean up?  Are we sure there aren't other references to this alert?
            # The user app probably has a reference to it but hopefully they'll clean up that reference too...
            try:
                self._alerts.remove(alert)
            except KeyError:
                # This is okay as it probably just means that the alert has already been cancelled by another thread...
                pass


    class AlertContext(object):
        """
        This object manages the life-cycle of an alert by ensuring all subscribers to the given alert topic are
        eventually reached: unreached subscribers are re-tried after some timeout using a different MDMT.
        This object contains little actual logic; it is more for book-keeping.
        """

        def __init__(self, msg, topic, subscribers, mdmts, _id):
            """
            Initiate the context object, which will simply store/manage which subscribers have been reached,
            which MDMTs have been used, and where to route responses.
            :param msg: the alert message being sent (raw string/bytes)
            :param topic: the topic of the alert
            :param subscribers: current subscribers to the alert at the time it was created
            :type subscribers: list
            :param mdmts: MDMTs available for sending this alert at the time of its creation
            :type mdmts: list
            :param _id: numeric unique ID for this alert
            """

            self.msg = msg
            self.topic = topic
            # create a copy to ignore later additions; use set to easily determine which ones we haven't reached yet
            self.subscribers = set(subscribers)
            self.mdmts = mdmts
            # TODO: if we update the MDMTs, need some way of changing the available ones known by the context...
            self.id = _id

            # This data will be updated as alerts are sent/retried/responded to
            self.subscribers_reached = set()
            self.mdmts_used = []

            # track whether we should continue trying to contact subscribers to this alert or not
            self.active = True

            # Used by RideD to ensure that simultaneous updates to this object don't corrupt it
            self.thread_lock = Lock()

        def record_mdmt_used(self, mdmt):
            """
            Maintain an ordered record of each attempted MDMT so that we can easily break ties for choosing the next
            one by using an un-used (or not-recently-used) option.
            :param mdmt:
            :return:
            """
            # THREADING: we don't bother locking since this should be called serially only from main send_alert or re-tx thread
            self.mdmts_used.append(mdmt)

        def is_mdmt_used(self, mdmt):
            """
            :return: True if the specified MDMT has been used during the lifetime of this alert
            """
            assert mdmt in self.mdmts, "requested MDMT %s not in the original list of available ones!" % mdmt
            return mdmt in self.mdmts_used

        def least_recently_used_mdmt(self):
            """
            :return: The least recently used MDMT if all have been used, otherwise the first one that hasn't
            """
            # THREADING: we don't bother locking since this should be called serially only from main send_alert or
            # re-tx thread; also it's basically read-only and the for loop uses a copy of the MDMTs.

            # First, check if we perhaps haven't used them all yet...
            for m in self.mdmts:
                if not self.is_mdmt_used(m):
                    return m

            # All have been used, so we can work our way backwards in those recently used until we account for all of
            # them, returning the last one seen as the least-recently used
            accounted_for = set()
            for m in reversed(self.mdmts):
                accounted_for.add(m)
                if len(accounted_for) == len(self.mdmts):
                    return m

        def most_recently_used_mdmts(self):
            """
            Returns up to k-1 (k=#MDMTs) of the most-recently-used MDMTs for use in selecting a different one to try next.
            Note: If none have been tried or all have been tried an equal number of times, this will return an empty list.
            :return:
            """
            # determine how many MDMTs have been selected the greatest # times and return them
            num_to_select = len(self.mdmts_used) % len(self.mdmts)
            if num_to_select:
                selected = self.mdmts_used[-num_to_select:]
                return selected
            return []

        def record_subscriber_reached(self, sub):
            with self.thread_lock:
                self.subscribers_reached.add(sub)

        def has_unreached_subscribers(self):
            return self.n_unreached_subscribers() > 0
        def n_unreached_subscribers(self):
            return len(self.unreached_subscribers())
        def unreached_subscribers(self):
            """
            :rtype: set
            :return:
            """
            return self.subscribers - self.subscribers_reached

        def __repr__(self):
            return "RideD.Alert#%d(topic=%s)" % (self.id, self.topic)
