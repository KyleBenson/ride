import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

import json
import networkx as nx


class SdnTopology(object):
    """Generates a networkx topology (undirected graph) from information
    gleaned from an SDN Controller.
    Supports various functions such as finding multicast spanning trees and
    installing flow rules.

    The inheritance hierarchy works like this: the base class implements
    most of the interesting algorithms by using various helper functions.
    The derived classes implement those helper functions in order to adapt
    a particular data model and API (e.g. SDN controller, generic graph, etc.)
    to the SdnTopology tool."""

    def __init__(self):
        super(SdnTopology, self).__init__()
        self.topo = nx.Graph()

    def build_topology(self):
        # TODO: refactor this to enable get_switches, get_hosts, get_links, etc? add_ funcs should return the component
        switches = self.rest_api.get_switches()
        log.debug("Switches: %s" % json.dumps(switches, sort_keys=True, indent=4))
        for s in switches:
            self.add_switch(s)

        # log.debug(self.topo.nodes())

        links = self.rest_api.get_links()
        log.debug("Links: %s" % json.dumps(links, sort_keys=True, indent=4))
        for link in links:
            self.add_link(link)

        # log.debug("Topo's edges before hosts: %s " % list(self.topo.edges(data=True)))

        hosts = self.rest_api.get_hosts()
        log.debug("Hosts: %s" % json.dumps(hosts, sort_keys=True, indent=4))
        for host in hosts:
            self.add_host(host)

        log.info("Final %d nodes: %s" % (self.topo.number_of_nodes(), list(self.topo.nodes(data=True))))
        log.info("Final %d edges: %s" % (self.topo.number_of_edges(), list(self.topo.edges(data=True))))

    ### Topoology-related generic methods: this is where the algorithms go!

    def get_multicast_tree(self, source, destinations):
        """Uses networkx algorithms to build a multicast tree for the given source node and
        destinations (an iterable).  Can be used to build and install flow rules."""

        try:
            from networkx.algorithms.approximation import steiner_tree
        except ImportError:
            raise NotImplementedError("Steiner Tree algorithm not found!")

        # we don't care about directionality of the mcast tree here,
        # so we can treat the source as yet another destination
        destinations.append(source)
        return steiner_tree(self.topo, destinations)

    def get_path(self, source, destination):
        """Gets shortest path by weight attribute between the nodes.
        @:return a sequence of nodes representing the shortest path"""

        return nx.shortest_path(self.topo, source=source, target=destination)

    # Generic flow rule generating functions based on the topology

    def get_flow_rules_from_path(self, path):
        """Converts a simple path to a list of flow rules that can then
        be installed in the corresponding switches.  The flow rules simply
        match based in in_port."""

        # can't just iterate over container as the
        # next/prev node is important for flow rules
        # as well as the switch they'll be applied to

        rules = []
        for src, switch, dst in zip(path[:-2], path[1:-1], path[2:]):
            # Since the edges in the topology are non-directional, we
            # need to determine which side of the links the src/dst are
            in_port, _ = self.get_ports_for_nodes(switch, src)
            out_port, _ = self.get_ports_for_nodes(switch, dst)

            actions = self.get_actions(("output", out_port))
            matches = self.get_matches(in_port=in_port)

            rules.append(self.get_flow_rule(switch, matches, actions))
        return rules

    def get_flow_rules_from_multicast_tree(self, tree, source, matches, group_id='1'):
        """Converts a multicast tree to a list of flow rules that can then
        be installed in the corresponding switches.  They will be ordered
        with group flows first so iterating over the list to install them
        should not cause BAD_OUT_GROUP errors.

        @:param tree - a networkx Graph-like object representing the multicast tree
        @:param source - source node/switch from which to start the search
        @:param matches - match rules to be used for matching multicast packets
        @:param group_id - group_id to assign this group rule to on all switches

        @:param flows - list of all flow rules to accomplish the multicast tree"""

        group_flows = []
        flows = []

        # Since we assume this is a directed multicast tree from the source,
        # we should traverse the tree in a specific order from that source.
        # Starting from the source host's switch (the host doesn't get flow rules),
        # look at each next node the tree reaches and install the proper flow rules for it.

        # When we encounter a leaf of the tree (i.e. a host receiving the multicast packet),
        # we convert the destination IP/MAC addresses to the final host's actual
        # IP/MAC address in order to avoid having to manage multicast addresses
        # being listened to on that host (MAC is necessary or it will drop packet).
        def __get_action(_node, _succ):
            port = self.get_ports_for_nodes(_node, _succ)[0]
            if self.is_host(_succ):
                _action = self.get_actions(("set_ipv4_dst", self.get_ip_address(_succ)),
                                           ("set_eth_dst", "ff:ff:ff:ff:ff:ff"),
                                           ("output", port))
            else:
                _action = self.get_actions(("output", port))
            return _action

        bfs = nx.bfs_successors(tree, source)
        bfs.next()  # skip source host
        for node, successors in bfs:
            # if only one successor, we don't need a group flow
            use_group_flow = len(successors) != 1
            if use_group_flow:
                # TODO: could move this to a helper function
                buckets = []
                for i, succ in enumerate(successors):
                    action = __get_action(node, succ)
                    buckets.append(self.get_bucket(action))
                group_flows.append(self.get_group_flow_rule(node, buckets, group_id, 'all'))
                action = self.get_actions(("group", group_id))
            else:
                action = __get_action(node, successors[0])

            # TODO: update matches with the src port/IP?

            flows.append(self.get_flow_rule(node, matches, action))

            # print node, successors

        group_flows.extend(flows)
        return group_flows

    ### Utility helper functions that must be implemented by base classes

    # Topology-related helper functions
    def add_link(self, link):
        """Adds the given link, in its raw input format, to the topology."""
        raise NotImplementedError
        # Implementation note: because of the undirected graph, ports need to
        # include the DPID (data plane ID) of the node like this:
        # self.topo.add_edge(link['src-switch'], link['dst-switch'], latency=link['latency'],
        #                    port1={'dpid': link['src-switch'], 'port_num': link['src-port']},
        #                    port2={'dpid': link['dst-switch'], 'port_num': link['dst-port']})

    def add_switch(self, switch):
        """Adds the given switch, in its raw input format, to the topology."""
        raise NotImplementedError
        # Usually pretty straightforward and just requires the node name:
        # self.topo.add_node(switch['id'])

    def add_host(self, host):
        """Adds the given host, in its raw input format, to the topology."""
        raise NotImplementedError

    def is_host(self, node):
        """Returns True if the given node is a host, False if it is a switch."""
        raise NotImplementedError

    def get_ip_address(self, host):
        """Gets the IP address associated with the given host in the topology."""
        ip = self.topo.node[host]['ip']
        if ip is None:
            raise AttributeError("Host %s has no IPv4 address!  Did you 'pingall 10' in Mininet?  Or maybe you need to override this method?" % self.topo.node[host])
        return ip

    def get_ports_for_nodes(self, n1, n2):
        """Returns a pair of port numbers (or IDs) corresponding with the link
        connecting the two specified nodes respectively.  More than one link
        connecting the nodes is undefined behavior."""

        # Because of the undirected graph model, we have to disambiguate
        # the directionality of the request in order to properly order
        # the return values.

        edge = self.topo[n1][n2]
        if edge['port1']['dpid'] == n1:
            port1 = edge['port1']['port_num']
            port2 = edge['port2']['port_num']
        else:
            port1 = edge['port2']['port_num']
            port2 = edge['port1']['port_num']

        return port1, port2

    # Flow rule helper functions

    def install_flow_rule(self, rule):
        """Helper function that assumes the data plane device (switch)
        to which this rule will be pushed is included in the rule object."""
        log.debug("Installing flow %s" % rule)
        return self.rest_api.push_flow_rule(rule)

    def install_group(self, group):
        """Helper function that assumes the data plane device (switch)
        to which this rule will be pushed is included in the rule object."""
        log.debug("Installing group %s" % group)
        return self.rest_api.push_group(group)

    def get_flow_rule(self, switch, matches, actions, **kwargs):
        """Builds a flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param matches - dict<str,str> of matches this flow performs as formatted by get_matches(...)
        @:param actions - str of OpenFlow actions to be taken as formatted by get_actions(...)
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()

        @:return rule - dict representing the flow rule that can be installed on the switch"""
        raise NotImplementedError

    def get_matches(self, **kwargs):
        """Properly format (for the particular controller) OpenFlow
        packet matching criteria and return the result.

        @:param **kwargs - common OpenFlow matches including, at a minimum,
         in_port, eth_type, ip_proto, <eth|ipv4|ipv6|tcp|udp>_<src|dst>,
         (use these exact spellings as a user, but be prepared to convert
         them to your controller's proper spelling if deriving this class)
        @:return matches - object representing matching criteria
        that can be passed to get_flow_rule(...) in order to format it
        properly for REST API

        NOTE: default implementation assumes we can simply return kwargs
        """
        return kwargs

    def get_actions(self, *args):
        """Properly format (for the particular controller) OpenFlow
        packet actions and return the result.

        @:param *args - ordered list of common OpenFlow actions in the form of either
         strings (actions with no arguments) or tuples
         where the first element is the action and the remaining elements are arguments.
         You should support, at a minimum, output, group, set_<eth|ipv4|ipv6>_<src|dst>
         (use these exact spellings as a user, but be prepared to convert
         them to your controller's proper spelling if deriving this class)
        """
        raise NotImplementedError

    def get_group_flow_rule(self, switch, buckets, group_id='1', group_type='all', **kwargs):
        """Builds a group flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param buckets - list of buckets where each bucket is formatted as returned from get_bucket(...)
        @:param type - type of group (all, indirect, select, fast_failover); defaults to 'all'
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()

        @:return rule - dict representing the flow rule that can be installed on the switch"""
        raise NotImplementedError

    def get_bucket(self, actions, weight=None, watch_group=None, watch_port=None):
        """Formats a dict-like object to use as a bucket within get_group_flow_rule.

        @:param actions - actions to perform
        @:param weight
        @:param watch_group
        @:param watch_port

        @:return bucket - dict representing the bucket with all necessary fields filled
        """
        raise NotImplementedError

    def __get_flow_rule(self, switch, **kwargs):
        """Helper function to assemble fields of a flow common between flow entry types.
        In particular, it should fill any fields that are REQUIRED by the controller's REST API.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()
        """
        raise NotImplementedError
