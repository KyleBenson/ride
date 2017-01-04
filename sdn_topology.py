import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO)

from floodlight_api import RestApi
import json
import networkx as nx

# Otherwise, identify by ipv4
IDENTIFY_HOSTS_BY_MAC = True


class SdnTopology(object):
    """Generates a networkx topology from information gleaned from an SDN Controller.
    Supports various functions such as finding multicast spanning trees and
    installing flow rules."""

    def __init__(self, ip='localhost', port='8080'):
        super(SdnTopology, self).__init__()

        self.rest_api = RestApi(ip, port)
        self.topo = nx.Graph()
        self.unique_counter = 0  # used for e.g. flow entry names

        self.build_topology()

    def build_topology(self):

        # TODO: move these into actual RestApi function calls
        cmd = 'switches'
        args = []

        path = self.rest_api.lookup_path(cmd, args)
        switches = json.loads(self.rest_api.get(path))

        log.debug("Switches: %s" % json.dumps(switches, sort_keys=True, indent=4))

        for s in switches:
            self.topo.add_node(s['switchDPID'])

        # log.debug(self.topo.nodes())


        cmd = 'link'
        path = self.rest_api.lookup_path(cmd, args)
        links = json.loads(self.rest_api.get(path))

        log.debug("Links: %s" % json.dumps(links, sort_keys=True, indent=4))

        for link in links:
            self.topo.add_edge(link['src-switch'], link['dst-switch'], latency=link['latency'],
                               port1={'dpid': link['src-switch'], 'port_num': link['src-port']},
                               port2={'dpid': link['dst-switch'], 'port_num': link['dst-port']})

        # log.debug("Topo's edges before hosts: %s " % list(self.topo.edges(data=True)))

        cmd = 'hosts'
        path = self.rest_api.lookup_path(cmd, args)
        hosts = json.loads(self.rest_api.get(path))

        log.debug("Hosts: %s" % json.dumps(hosts, sort_keys=True, indent=4))

        for host in hosts['devices']:
            # assume hosts only have a single IP address/interface/MAC address/attachment point,
            # but potentially multiple VLANs
            mac = ipv4 = None

            try:
                switch = host['attachmentPoint'][0]
            except IndexError:
                log.debug("Skipping host with no attachmentPoint: %s" % host)
                continue

            try:
                mac = host['mac'][0]
                ipv4 = host['ipv4'][0]
            except IndexError:
                if mac is None:
                    log.debug("Skipping host with no MAC or IPv4 addresses: %s" % host)
                    continue

            if IDENTIFY_HOSTS_BY_MAC:
                hostid = mac
            else:
                hostid = ipv4

            self.topo.add_node(hostid, mac=mac, vlan=host['vlan'], ipv4=ipv4)
            # TODO: generalize the creation of an edges' attributes? maybe similar for adding a host?
            self.topo.add_edge(hostid, switch['switch'], latency=0,
                               port1={'dpid': hostid, 'port_num': 0},
                               # for some reason, REST API gives us port# as a str
                               port2={'dpid': switch['switch'], 'port_num': int(switch['port'])})

        log.info("Final %d nodes: %s" % (self.topo.number_of_nodes(), list(self.topo.nodes())))
        log.info("Final %d edges: %s" % (self.topo.number_of_edges(), list(self.topo.edges(data=True))))

    # Utility helper functions

    @staticmethod
    def is_host(node):
        """Returns True if the given node is a host, False if it is a switch.
        Does NOT determine if the node is in the topology."""

        # Switches have more bits in 'MAC address' and
        # are never identified by IP address
        if node.count(':') == 7 and node.count('.') == 0:
            return False
        else:
            return True

    def get_multicast_tree(self, source, destinations):
        """Uses networkx algorithms to build a multicast tree for the given source node and
        destinations (an iterable).  Can be used to build and install flow rules."""

        try:
            from networkx.algorithms.approximation import steiner_tree
        except ImportError:
            raise NotImplementedError("Steiner Tree algorithm not found!")

        # we don't care about directionality of the mcast tree here
        destinations.append(source)
        return steiner_tree(self.topo, destinations)

    def get_path(self, source, destination):
        """Gets shortest path by weight attribute between the nodes.
        @:return a sequence of nodes representing the shortest path"""

        return nx.shortest_path(self.topo, source=source, target=destination)

    def get_flow_rule(self, switch, matches, actions, **kwargs):
        """Builds a flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param matches - dict<str,str> of matches this flow performs
        @:param actions - str of OpenFlow actions to be taken e.g. 'strip_vlan,output=3'
        @:param **kwargs - all remaining kwargs are added to the flow rule dict

        @:return rule - dict representing the flow rule that can be installed on the switch"""

        rule = self.__get_flow_rule(switch, **kwargs)
        rule['actions'] = actions
        rule.update(matches)

        return rule

    def get_bucket(self, bucket_id, actions):
        """Formats a dict-like object to use as a bucket within get_group_flow_rule.

        @:param bucket_id - an integer that uniquely identifies this bucket within the group (used for prioritizing)
        @:param actions - list of actions to perform

        @:return bucket - dict representing the bucket with all necessary fields filled
        """

        bucket = {'bucket_id': bucket_id,
                  'bucket_actions': actions,
                  # HACK: this is necessary due to a bug in Floodlight that
                  # sets the value to "any" when unspecified, causing an error
                  "bucket_watch_group": "any"
                  }
        # TODO: handle bucket_weight, bucket_watch_group, bucket_watch_port variables
        return bucket

    def get_group_flow_rule(self, switch, buckets, group_id='1', group_type='all', **kwargs):
        """Builds a group flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param buckets - list of buckets where each bucket is formatted as returned from get_bucket(...)
        @:param type - type of group (all, indirect, select, fast_failover); defaults to 'all'

        @:return rule - dict representing the flow rule that can be installed on the switch"""

        rule = self.__get_flow_rule(switch, **kwargs)
        rule['group_buckets'] = buckets
        rule['entry_type'] = 'group'
        rule['group_type'] = group_type
        rule['group_id'] = group_id
        return rule

    def __get_flow_rule(self, switch, **kwargs):
        """Helper function to assemble fields of a flow common between flow entry types.
        In particular, it assigns a unique name to the flow rule if you did not explicitly."""

        name = kwargs.get('name', None)
        if name is None:
            name = "anon_flow_%d" % self.unique_counter
            self.unique_counter += 1

        rule = {'switch': switch,
                "name": name,
                "active": "true",
                }
        return rule

    def install_flow_rule(self, rule):
        return self.rest_api.push_flow_rule(rule)

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
            in_edge = self.topo[src][switch]

            if in_edge['port1']['dpid'] == src:
                in_port = in_edge['port2']['port_num']
            else:
                in_port = in_edge['port1']['port_num']

            out_edge = self.topo[switch][dst]
            if out_edge['port1']['dpid'] == dst:
                out_port = out_edge['port2']['port_num']
            else:
                out_port = out_edge['port1']['port_num']

            actions = "output=%d" % out_port
            matches = {"in_port": str(in_port)}

            rules.append(self.get_flow_rule(switch, matches, actions))
        return rules

    def get_flow_rules_from_multicast_tree(self, tree, source=None):
        """Converts a multicast tree to a list of flow rules that can then
        be installed in the corresponding switches.

        @:param tree - a networkx Graph-like object representing the multicast tree
        @:param source - source node/switch from which to start the search (optional)"""

        # NOTE: can't just iterate over the edges in the tree as each
        # switch requires a group entry to be made as well

        # for multicast, we could probably get away with splitting up the group table entry (sources/switch)
        # from the buckets (switch/destinations)

        # can use get_flow_rule_for_link whenever it doesn't branch
        pass


#### Helper functions for tests

def mac_for_host(host_num):
    """Assuming you started mininet with --mac option, this returns a
    mac address for host h<host_num> e.g. for h1 do mac_for_host(1) --> 00:00:00:00:00:01"""

    # format int as hex with proper number of octets, then add :'s using some pymagic
    num = format(host_num, 'x').rjust(12, '0')
    num = ':'.join(s.encode('hex') for s in num.decode('hex'))
    return num


def dpid_for_switch(switch_num):
    """This returns a DPID for switch s<switch_num>
    e.g. for s1 do dpid_for_switch(1) --> 00:00:00:00:00:00:00:01"""

    # Assume we won't have enough switches to ever break this...
    return "00:00:%s" % mac_for_host(switch_num)


#### Tests


def test_path_flow(st):
    """Test simple static flow entries for a basic path between h1 and h16"""

    # path = st.get_path("10.0.0.1", "10.0.0.16")
    path = st.get_path(mac_for_host(1), mac_for_host(16))
    # print "Path:", path
    rules = st.get_flow_rules_from_path(path)
    rules.extend(st.get_flow_rules_from_path(list(reversed(path))))
    log.debug("Rules: %s" % rules)
    for rule in rules:
        st.install_flow_rule(rule)


def test_group_flow(st):
    switch = dpid_for_switch(1)
    matches = {'ipv4_src': '10.0.0.1',
               'eth_type': '0x0800'
               }
    actions = "group=1"

    flow = st.get_flow_rule(switch, matches, actions, priority=500)
    buckets = [st.get_bucket(1, 'output=3')]
    gflow = st.get_group_flow_rule(switch, buckets)
    # print flow
    # print gflow
    st.install_flow_rule(gflow)
    st.install_flow_rule(flow)

    #TODO: some API for doing both of the flows in one go


def test_mcast_flows(st):
    # mcast_tree = st.get_multicast_tree("10.0.0.1", ["10.0.0.2", "10.0.0.11", "10.0.0.16"])
    mcast_tree = st.get_multicast_tree(mac_for_host(1), [mac_for_host(2), mac_for_host(11), mac_for_host(16)])
    print list(mcast_tree.nodes())

    pass


if __name__ == '__main__':
    st = SdnTopology()

    # test_path_flow(st)
    # FIXME: we'll probably have to do the simple flows better (IP address too) in order to avoid conflicts

    test_group_flow(st)
    #test_mcast_flows(st)
