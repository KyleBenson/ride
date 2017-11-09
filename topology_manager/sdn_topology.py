import argparse
import logging
log = logging.getLogger(__name__)
import json
import networkx as nx

from network_topology import NetworkTopology
MAX_OPENFLOW_PRIORITY = 65535


class SdnTopology(NetworkTopology):
    """Generates a networkx topology (undirected graph) from information
    gleaned from an SDN Controller.
    Supports various functions such as finding multicast spanning trees and
    installing flow rules.

    The inheritance hierarchy works like this: the base class (NetworkTopology)
    implements the networking-related graph algorithms by using various helper functions.
    This class adds functionality for SDN/OpenFlow by maintaining the underlying
    topology with proper annotations for use by its helper functions that are
    used to generate flow rules for insertion into switches.
    Derived classes implement abstract helper functions in order to adapt
    a particular data model and API (e.g. SDN controller, generic graph, etc.)
    to the SdnTopology tool."""

    def __init__(self, rest_api):
        """
        :param rest_api.base_rest_api.BaseRestApi rest_api:
        :type rest_api: rest_api.base_rest_api.BaseRestApi
        """
        super(SdnTopology, self).__init__()
        self.rest_api = rest_api

    def build_topology(self, from_scratch=True):
        """
        Builds the topology by getting all the switches, links, and hosts from the underlying REST API and then
        incrementally adding each of those components to the topology in its implementation-specific manner.
        :param from_scratch: if True, discards the old topology first (NOTE: if you expect components to leave the network,
        you should specify True so that their lack of presence will be noted!)
        :return:
        """

        # NOTE: we gather up the raw values first and then add the components all at once to minimize the time
        # during which the topology is unstable (e.g. missing links between switches).

        # TODO: refactor this to enable get_switches, get_hosts, get_links, etc? add_ funcs should return the component
        switches = self.rest_api.get_switches()
        log.debug("Switches: %s" % json.dumps(switches, sort_keys=True, indent=4))

        # log.debug(self.topo.nodes())

        links = self.rest_api.get_links()
        log.debug("Links: %s" % json.dumps(links, sort_keys=True, indent=4))

        # log.debug("Topo's edges before hosts: %s " % list(self.topo.edges(data=True)))

        hosts = self.rest_api.get_hosts()
        log.debug("Hosts: %s" % json.dumps(hosts, sort_keys=True, indent=4))

        # TODO: add thread-safe version capability where we'll lock self.topo and release it after finishing the topology update
        # NOTE: this would need a reader-writer thread lock that allows multiple readers but only one writer if no other readers (last part would require prioritization)

        if from_scratch:
            self.topo.clear()

        # now add all the components
        for s in switches:
            self.add_switch(s)
        for link in links:
            self.add_link(link)
        for host in hosts:
            self.add_host(host)

        log.info("Final %d nodes: %s" % (self.topo.number_of_nodes(), list(self.topo.nodes(data=True))))
        log.info("Final %d edges: %s" % (self.topo.number_of_edges(), list(self.topo.edges(data=True))))

    @classmethod
    def get_arg_parser(cls):
        arg_parser = argparse.ArgumentParser(add_help=False)

        arg_parser.add_argument('--ip', default='127.0.0.1', dest='controller_ip',
                                help='''IP address of SDN controller we'll use (default=%(default)s)''')
        arg_parser.add_argument('--port', default=8181, type=int, dest='controller_port',
                                help='''port number of SDN controller's REST API that we'll use (default=%(default)s)''')
        arg_parser.add_argument('--topology-adapter', default='onos', dest='topology_adapter_type',
                                help='''type of SdnTopology to use as the SDN Controller adapter (default=%(default)s)''')

        return arg_parser

    # Generic flow rule generating functions based on the topology

    def build_flow_rules_from_path(self, path, use_matches=None, add_matches=None, **kwargs):
        """
        Converts a simple path to a list of flow rules that can then be installed in the corresponding switches.
        :param use_matches: optionally use the specified 'matches'; if unspecified, by default the flow rules simply
         match based on in_port, ipv4_src, and ipv4_dst.
        :param add_matches: an optional dict to be used as additional parameters (key-value pairs) to build_matches
         (NOTE: this cannot be used in conjunction with use_matches! BUT it CAN be used to overwrite the default
         matches i.e. ipv4_src/ipv4_dst/in_port!)
        :param kwargs: additional arguments passed to build_flow_rule()
        """

        # can't just iterate over container as the
        # next/prev node is important for flow rules
        # as well as the switch they'll be applied to

        src_ip = dst_ip = None
        if use_matches is None:
            src_ip = self.get_ip_address(path[0])
            dst_ip = self.get_ip_address(path[-1])

        rules = []
        for src, switch, dst in zip(path[:-2], path[1:-1], path[2:]):
            # Since the edges in the topology are non-directional, we
            # need to determine which side of the links the src/dst are
            in_port, _ = self.get_ports_for_nodes(switch, src)
            out_port, _ = self.get_ports_for_nodes(switch, dst)

            actions = self.build_actions(("output", out_port))

            if use_matches is None:
                if add_matches is None:
                    add_matches = dict()
                matches_params = dict(in_port=in_port, ipv4_src=src_ip, ipv4_dst=dst_ip)
                matches_params.update(add_matches)
                matches = self.build_matches(**matches_params)
            else:
                matches = use_matches

            rules.append(self.build_flow_rule(switch, matches, actions, **kwargs))
        return rules

    def build_flow_rules_from_multicast_tree(self, tree, source, matches, group_id='1', route_responses=False, **kwargs):
        """Converts a multicast tree to a list of flow rules that can then
        be installed in the corresponding switches.  They will be ordered
        with group flows first so iterating over the list to install them
        should not cause BAD_OUT_GROUP errors.

        NOTE: even though your matches might include a true multicast IP address, this implementation
        translates the packet to look like a unicast one (i.e. ipv4_dst matches that of the receiving host)
        before delivery to receivers.  A future version may allow you to selectively disable this packet manipulation...

        :param tree - a networkx Graph-like object representing the multicast tree
        :type tree: nx.Graph
        :param source - source node/switch from which to start the search
        :param matches - match rules to be used for matching multicast packets
        :param group_id - group_id to assign this group rule to on all switches
        :param route_responses: if True (is not by default), also build flow rules to route responses backwards
         along the tree (NOTE: uses the default 'matches' provided by build_flow_rules_from_path(p))
        :param kwargs: additional arguments passed to build_flow_rule()

        :return group_flows, flows - pair of list of all flow rules to accomplish the multicast tree"""

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
                _action = self.build_actions(("set_ipv4_dst", self.get_ip_address(_succ)),
                                             ("set_eth_dst", "ff:ff:ff:ff:ff:ff"),
                                             ("output", port))
            else:
                _action = self.build_actions(("output", port))
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
                    buckets.append(self.build_bucket(action))
                group_flows.append(self.build_group(node, buckets, group_id, 'ALL'))
                action = self.build_actions(("group", group_id))
            else:
                action = __get_action(node, successors[0])

            # TODO: update matches with the src port/IP?

            flows.append(self.build_flow_rule(node, matches, action, **kwargs))

            # print node, successors

        # To ensure responses flow along the same route as the multicast query, we offer this option to install static
        # routes in the reverse direction:
        if route_responses:
            # make sure we ignore switches!
            leaves = set(n for n in tree.nodes() if tree.degree(n) == 1) - {source}
            for node in leaves:
                path = nx.shortest_path(tree, node, source)
                # TODO: include the 'matches' param?
                response_flows = self.build_flow_rules_from_path(path)
                print 'adding response flows for node %s: %s' % (node, response_flows)
                flows.extend(response_flows)

        return group_flows, flows

    def build_redirection_flow_rules(self, source, old_dest, new_dest=None, route=None, tp_protocol=None,
                                     source_port=None, old_dest_port=None, new_dest_port=None, **kwargs):
        """
        Re-directs a packet by translating the destination IP/ethernet address and port numbers to match that of the
        new_dest.  The source and old_dest are used for creating the matches part of the flow rules.  Flow rules for
        the route from source to new_dest will be added; they will match the optionally-specified route.  The flow rule
        that handles packet modification for the new addresses will be installed on the first switch along route.
        The flow rules also include routing/translations for the opposite direction (i.e. new_dest --> source) in
        order to ensure the source host seamlessly receives responses (e.g. ACKs) properly.  Specifying the port
        numbers is highly recommended to ensure these translations don't prevent other applications than the intended
        one running on the same hosts from communicating with each other.

        FUTURE ENHANCE: a 'matches' parameter could be added so that you can specify how the match is done (this would
        make the source/old_dest arguments potentially optional);
        a 'switch' parameter may be added so that you can specify which switch on the route does the packet modification

        :param source:
        :param old_dest: the old destination host from which we'll extract the IP/ethernet addresses for matching
        :param new_dest: the new destination the packet will be forwarded to; if unspecified, route[-1] will be assumed
        :param route: optional route that will be followed for the redirection; if unspecified, one will be chosen using
         self.get_path(source, new_dest)
        :param tp_protocol: one of 'tcp', 'udp', or 'sctp' to specify the transport-layer protocol; MUST be specified if
        any of the ports are!
        :param source_port: optional port # that will be added to matches & translation actions if specified
        :param old_dest_port: optional port # that will be added to matches & translation actions if specified
        :param new_dest_port: optional port # that will be added to matches & translation actions if specified
        :param kwargs: additional arguments passed to build_flow_rule()
        :return: a list of flow rules for accomplishing the requested redirection routing
        """

        if route is None:
            if new_dest is None:
                raise ValueError("You must at least specify the new_dest argument to determine the redirection path!")
            route = self.get_path(source, new_dest)
        elif new_dest is None:
            new_dest = route[-1]

        assert source == route[0] and new_dest == route[-1],\
            "redirection route requested that didn't match requested source and destination! Src/dst: %s/%s\n" \
            "Requested Route: %s" % (source, new_dest, route)

        if tp_protocol is None and (source_port is not None or old_dest_port is not None or new_dest_port is not None):
            raise ValueError("if you specify one of the port numbers for redirection you MUST specify the tp_protocol!")
        if tp_protocol is not None and tp_protocol not in ('tcp', 'udp', 'sctp'):
            raise ValueError("unrecognized transport protocol type: %s" % tp_protocol)

        flow_rules = []
        # We'll custom make the packet manipulation flow rule for the first switch in the path
        src_ip = self.get_ip_address(source)
        src_eth = self.get_mac_address(source)
        old_dest_ip = self.get_ip_address(old_dest)
        old_dest_eth = self.get_mac_address(old_dest)
        switch = route[1]
        in_port = self.get_ports_for_nodes(source, switch)[1]

        match_params = dict(ipv4_src=src_ip, ipv4_dst=old_dest_ip, in_port=in_port,
                            eth_src=src_eth, eth_dst=old_dest_eth)
        if source_port is not None:
            match_params[tp_protocol + '_src'] = source_port
        if old_dest_port is not None:
            match_params[tp_protocol + '_dst'] = old_dest_port
        # Must do this if we want to alter the transport layer port #!
        if tp_protocol is not None:
            match_params['ip_proto'] = tp_protocol
        matches = self.build_matches(**match_params)

        new_dest_ip = self.get_ip_address(new_dest)
        new_dest_eth = self.get_mac_address(new_dest)
        out_port = self.get_ports_for_nodes(switch, route[2])[0]
        # NOTE: we have to do an output action as some REST APIs (e.g. ONOS) don't support redirect, or table(0)...
        actions = [("set_eth_dst", new_dest_eth), ("set_ipv4_dst", new_dest_ip), ("output", out_port)]
        if new_dest_port is not None:
            actions.insert(-1, ('set_%s_dst' % tp_protocol, new_dest_port))
        actions = self.build_actions(*actions)

        flow_rules.append(self.build_flow_rule(switch, matches, actions, **kwargs))

        # Now we'll use our helper methods to build the remaining flow rules, but ignore the one for the first switch.
        # We can do this without any further modifications since we already translated the packet's addresses/ports.
        add_matches = dict(eth_src=src_eth, eth_dst=new_dest_eth)
        if source_port is not None:
            add_matches[tp_protocol + '_src'] = source_port
        if new_dest_port is not None:
            add_matches[tp_protocol + '_dst'] = new_dest_port
        # match based on old port if we didn't translate it
        elif old_dest_port is not None:
            add_matches[tp_protocol + '_dst'] = old_dest_port
        other_rules = self.build_flow_rules_from_path(route, add_matches=add_matches if add_matches else None, **kwargs)
        # XXX: we assume our implementation of this method remains consistent (flow rule list is in same order as route)
        flow_rules.extend(other_rules[1:])

        # Now, we need to install rules so that when new_dest replies to source the packets will be translated to look
        # like they came from old_dest.  This requires more exact matching (e.g. ports) to ensure this only affects
        # e.g. one application rather than ALL traffic between them!
        #
        # Again, we first make a custom packet manipulation rule at the first switch from the 'source' (i.e. new_dest),
        # but this time we're translating the source address!
        rev_old_source_ip = new_dest_ip
        rev_new_source_ip = old_dest_ip
        rev_dest_ip = src_ip
        rev_old_source_port = new_dest_port
        rev_new_source_port = old_dest_port
        rev_dest_port = source_port
        rev_old_source_eth = new_dest_eth
        rev_new_source_eth = old_dest_eth
        rev_dest_eth = src_eth

        route = list(reversed(route))
        switch = route[1]
        in_port = self.get_ports_for_nodes(new_dest, switch)[1]

        match_params = dict(ipv4_src=rev_old_source_ip, ipv4_dst=rev_dest_ip, in_port=in_port,
                            eth_src=rev_old_source_eth, eth_dst=rev_dest_eth)
        if rev_old_source_port is not None:
            match_params[tp_protocol + '_src'] = rev_old_source_port
        if rev_dest_port is not None:
            match_params[tp_protocol + '_dst'] = rev_dest_port
        # Must do this if we want to alter the transport layer port #!
        if tp_protocol is not None:
            match_params['ip_proto'] = tp_protocol
        matches = self.build_matches(**match_params)

        out_port = self.get_ports_for_nodes(switch, route[2])[0]
        # NOTE: we have to do an output action as some REST APIs (e.g. ONOS) don't support redirect, or table(0)...
        actions = [("set_eth_src", rev_new_source_eth), ("set_ipv4_src", rev_new_source_ip), ("output", out_port)]
        if rev_new_source_port is not None:
            # Be careful with the ordering here!
            actions.insert(-1, ('set_%s_src' % tp_protocol, rev_new_source_port))
        actions = self.build_actions(*actions)

        flow_rules.append(self.build_flow_rule(switch, matches, actions, **kwargs))

        # Now we'll use our helper methods to build the remaining flow rules, but ignore the one for the first switch.
        # We can do this without any further modifications since we already translated the packet's addresses/ports.
        # NOTE: we have to overwrite the ipv4_src match rule since we changed it from what's seen on 'route'!
        add_matches = dict(ipv4_src=rev_new_source_ip, eth_src=rev_new_source_eth, eth_dst=rev_dest_eth)

        if rev_new_source_port is not None:
            add_matches[tp_protocol + '_src'] = rev_new_source_port
        # match based on old port if new isn't specified
        elif rev_old_source_port is not None:
            add_matches[tp_protocol + '_src'] = rev_old_source_port
        if rev_dest_port is not None:
            add_matches[tp_protocol + '_dst'] = rev_dest_port
        other_rules = self.build_flow_rules_from_path(route, add_matches=add_matches, **kwargs)
        # XXX: we assume our implementation of this method remains consistent (flow rule list is in same order as route)
        flow_rules.extend(other_rules[1:])

        return flow_rules

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

    # Host helper functions

    def is_host(self, node):
        """Returns True if the given node is a host, False if it is a switch."""
        raise NotImplementedError

    def get_hosts(self, attributes=False):
        return [n for n in self.topo.nodes(data=attributes) if self.is_host(n[0] if attributes else n)]

    def get_host(self, host):
        return self.topo.node[host]

    def get_host_by_ip(self, ip):
        candidates = [h for h in self.get_hosts() if self.get_ip_address(h) == ip]
        if len(candidates) > 1:
            raise ValueError("Found multiple hosts with IP address %s" % ip)
        return candidates[0]

    def get_host_by_mac(self, mac):
        candidates = [h for h in self.get_hosts() if self.get_mac_address(h) == mac]
        if len(candidates) > 1:
            raise ValueError("Found multiple hosts with MAC address %s" % mac)
        return candidates[0]

    def get_mac_address(self, host):
        """Gets the MAC address associated with the given host in the topology."""
        mac = self.get_host(host)['mac']
        if mac is None:
            raise AttributeError("Host %s has no MAC address!  Did you do ARP?  Or maybe you need to override this method?" % self.topo.node[host])
        return mac

    def get_ip_address(self, host):
        """Gets the IP address associated with the given host in the topology."""
        h = None
        try:
            h = self.get_host(host)
            ip = h['ip']
        except KeyError:
            raise AttributeError("Host %s has no IPv4 address, just %s!  Did you 'pingall 10' in Mininet?  Or maybe you need to override this method?" % (host, h))
        return ip

    def get_ports_for_nodes(self, n1, n2):
        """Returns a pair of port numbers (or IDs) corresponding with the link
        connecting the two specified nodes respectively.  More than one link
        connecting the nodes is undefined behavior."""

        # Because of the undirected graph model, we have to disambiguate
        # the directionality of the request in order to properly order
        # the return values.

        assert not isinstance(self.topo, nx.DiGraph), "We assume the graph is undirected, but self.topo is a nx.DiGraph!"
        edge = self.topo[n1][n2]
        if edge['port1']['dpid'] == n1:
            port1 = edge['port1']['port_num']
            port2 = edge['port2']['port_num']
        else:
            port1 = edge['port2']['port_num']
            port2 = edge['port1']['port_num']

        return port1, port2

    # Switch helper functions
    def is_switch(self, switch):
        """By default, we assume a non-host node is a switch."""
        return not self.is_host(switch)

    def get_switches(self, attributes=False):
        return [n for n in self.topo.nodes(data=attributes) if self.is_switch(n[0] if attributes else n)]

    def get_switch(self, switch):
        return self.topo.node[switch]

    # Flow rule helper functions

    def install_flow_rule(self, rule):
        """Helper function that assumes the data plane device (switch)
        to which this rule will be pushed is included in the rule object."""
        log.debug("Installing flow %s" % rule)
        return self.rest_api.push_flow_rule(rule)

    def install_flow_rules(self, rules):
        """Helper function for installing multiple flow rules, which simply iterates over them to call
         self.install_flow_rule(rule) and returns a list of return values.  Override this method to handle batch
         flow rule installation methods if your rest_api handles it!"""

        ret = []
        for fr in rules:
            ret.append(self.install_flow_rule(fr))
        return ret

    def get_flow_rules(self, switch=None):
        """
        Get all flow rules (optionally only those associated with switch).
        :param switch:
        :return:
        """
        return self.rest_api.get_flow_rules(switch)

    def get_groups(self, switch=None):
        """
        Get all groups (optionally only those associated with switch).
        :param switch:
        :return:
        """
        return self.rest_api.get_groups(switch)

    def remove_flow_rule(self, switch_id, flow_id):
        return self.rest_api.remove_flow_rule(switch_id, flow_id)

    def remove_all_flow_rules(self):
        """
        Removes all flow rules from all managed devices that have been added using the REST API.
        :return:
        """
        return self.rest_api.remove_all_flow_rules()

    def remove_all_groups(self, switch_id=None):
        """Remove all groups or optionally all groups from the specified switch."""
        return self.rest_api.remove_all_groups(switch_id)

    def install_group(self, group):
        """Helper function that assumes the data plane device (switch)
        to which this rule will be pushed is included in the rule object."""
        log.debug("Installing group %s" % group)
        return self.rest_api.push_group(group)

    def build_flow_rule(self, switch, matches, actions, **kwargs):
        """Builds a flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param matches - dict<str,str> of matches this flow performs as formatted by build_matches(...)
        @:param actions - str of OpenFlow actions to be taken as formatted by build_actions(...)
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()

        @:return rule - dict representing the flow rule that can be installed on the switch"""
        raise NotImplementedError

    def build_matches(self, **kwargs):
        """Properly format (for the particular controller) OpenFlow
        packet matching criteria and return the result.

        @:param **kwargs - common OpenFlow matches including, at a minimum,
         in_port, eth_type, ip_proto, <eth|ipv4|ipv6|tcp|udp>_<src|dst>,
         (use these exact spellings as a user, but be prepared to convert
         them to your controller's proper spelling if deriving this class)
        @:return matches - object representing matching criteria
        that can be passed to build_flow_rule(...) in order to format it
        properly for REST API

        NOTE: default implementation assumes we can simply return kwargs
        after verifying that eth_type is present (and properly set) if
        either ipv4 or ipv6 are used.  Similarly, we will have to set
        the ip_proto if a transport-layer IP protocol (e.g. UDP/TCP)
        are requested.
        """

        # need to check individual keys as they might be e.g. ipv4_dst
        if any('ipv4' in k for k in kwargs.keys()) and 'eth_type' not in kwargs:
            kwargs['eth_type'] = '0x0800'
        elif any('ipv6' in k for k in kwargs.keys()) and 'eth_type' not in kwargs:
            kwargs['eth_type'] = '0x86DD'

        # now for transport-layer
        # NOTE: this has only been tested for ONOS!
        key = 'ip_proto'
        values = dict(udp=17, tcp=6, sctp=132)
        if any('udp' in k for k in kwargs.keys()) and key not in kwargs:
            kwargs[key] = values['udp']
        elif any('tcp' in k for k in kwargs.keys()) and key not in kwargs:
            kwargs[key] = values['tcp']
        elif any('sctp' in k for k in kwargs.keys()) and key not in kwargs:
            kwargs[key] = values['sctp']
        elif key in kwargs and not isinstance(kwargs[key], int):
            try:
                kwargs[key] = values[kwargs[key]]
            except KeyError:
                raise ValueError("unrecognized ip_proto type '%s'" % kwargs[key])

        return kwargs

    def build_actions(self, *args):
        """Properly format (for the particular controller) OpenFlow packet actions and return the result.  Note that
        you must be careful with the ordering of actions!  For example, you typically want to have the 'output' action
        come last after any header modifications.  Failure to do so will output the packet and THEN modify the header!

        @:param *args - ordered list of common OpenFlow actions in the form of either
         strings (actions with no arguments) or tuples
         where the first element is the action and the remaining elements are arguments.
         You should support, at a minimum, output, group, set_<eth|ipv4|ipv6>_<src|dst>
         (use these exact spellings as a user, but be prepared to convert
         them to your controller's proper spelling if deriving this class)
        """
        raise NotImplementedError

    def build_group(self, switch, buckets, group_id='1', group_type='all', **kwargs):
        """Builds a group flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param buckets - list of buckets where each bucket is formatted as returned from build_bucket(...)
        @:param group_id - ID of the group (specified as a string), with default of '1'
        @:param type - type of group (all, indirect, select, fast_failover); defaults to 'all'
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()

        @:return rule - dict representing the group that can be installed on the switch"""
        raise NotImplementedError

    def build_bucket(self, actions, weight=None, watch_group=None, watch_port=None):
        """Formats a dict-like object to use as a bucket within build_group.

        @:param actions - actions to perform
        @:param weight
        @:param watch_group
        @:param watch_port

        @:return bucket - dict representing the bucket with all necessary fields filled
        """
        raise NotImplementedError

    def __build_flow_rule(self, switch, **kwargs):
        """Helper function to assemble fields of a flow common between flow entry types.
        In particular, it should fill any fields that are REQUIRED by the controller's REST API.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()
        """
        raise NotImplementedError
