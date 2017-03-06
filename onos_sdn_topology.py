import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

from onos_api import OnosRestApi
from sdn_topology import SdnTopology


class OnosSdnTopology(SdnTopology):
    """Generates a networkx topology from information gleaned from an SDN Controller.
    Supports various functions such as finding multicast spanning trees and
    installing flow rules."""

    def __init__(self, ip='localhost', port='8181'):
        super(OnosSdnTopology, self).__init__()

        self.rest_api = OnosRestApi(ip, port)

        self.build_topology()

    # Topology management helper functions

    def add_link(self, link):
        """Adds the given link, in its raw input format, to the topology."""
        # Link looks like: {"src":{"port":"5","device":"of:0000000000000002"},"dst":{"port":"1","device":"of:0000000000000001"},"type":"DIRECT","state":"ACTIVE"}
        # NOTE: ONOS lists links for both directions,
        # but networkx handles this gracefully by ignoring a duplicate.
        self.topo.add_edge(link['src']['device'], link['dst']['device'], state=link['state'],
                           port1={'dpid': link['src']['device'], 'port_num': int(link['src']['port'])},
                           port2={'dpid': link['dst']['device'], 'port_num': int(link['dst']['port'])})

    def add_switch(self, switch):
        """Adds the given switch, in its raw input format, to the topology."""
        # Device looks like: {"id":"of:0000000000000003","type":"SWITCH","available":true,"role":"MASTER","mfr":"Nicira, Inc.","hw":"Open vSwitch","sw":"2.6.90","serial":"None","chassisId":"3","annotations":{"managementAddress":"127.0.0.1","protocol":"OF_13","channelId":"127.0.0.1:42917"}}
        self.topo.add_node(switch['id'])

    def add_host(self, host):
        """Adds the given host, in its raw input format, to the topology."""
        # Host looks like: {"id":"00:00:00:00:00:09/None","mac":"00:00:00:00:00:09","vlan":"None","ipAddresses":["10.0.0.9"],"location":{"elementId":"of:0000000000000004","port":"1"}}

        # assume hosts only have a single IP address/interface/MAC address/attachment point,
        # but potentially multiple VLANs.
        # ONOS seems to assume only potential multiples of VLAN and IP address.
        mac = ip = None

        try:
            switch = host['location']['elementId']
            port = int(host['location']['port'])
        except IndexError:
            log.debug("Skipping host with no location in topology: %s" % host)
            return

        try:
            mac = host['mac']
            ip = host['ipAddresses'][0]
        except IndexError:
            if mac is None:
                log.debug("Skipping host with no MAC or IPv4 addresses: %s" % host)
                return

        self.topo.add_node(host['id'], mac=mac, vlan=host['vlan'], ip=ip)
        self.topo.add_edge(host['id'], switch,
                           port1={'dpid': host['id'], 'port_num': 0},
                           port2={'dpid': switch, 'port_num': port})

    def is_host(self, node):
        """Returns True if the given node is a host, False if it is a switch.
        Does NOT determine if the node is in the topology."""

        # Switches have more bits in 'MAC address' and
        # are never identified by IP address
        if node.count(':') == 5:
            return True
        elif node.startswith("of:"):
            return False
        else:
            raise NameError

    # Flow rule helper functions

    def get_flow_rule(self, switch, matches, actions, **kwargs):
        """Builds a flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param matches - dict<str,str> of matches this flow performs
        @:param actions - str of OpenFlow actions to be taken e.g. 'strip_vlan,output=3'
        @:param **kwargs - all remaining kwargs are added to the flow rule dict

        @:return rule - dict representing the flow rule that can be installed on the switch"""

        rule = self.__get_flow_rule(switch, **kwargs)
        rule['treatment'] = {'instructions': actions}
        rule['selector'] = {'criteria': matches}
        return rule

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
        """

        # ONOS uses a weird REST API format for matches in which they
        # include a key name for the value specified.  Hence, we need
        # to look up this key name in a dict in order to build
        # the resulting flow match dict correctly.
        key_names = {u'CHANNEL_SPACING': u'channelSpacing',
                     u'ETH_DST': u'mac',
                     u'ETH_SRC': u'mac',
                     u'ETH_TYPE': u'ethType',
                     u'ICMPV4_CODE': u'icmpCode',
                     u'ICMPV4_TYPE': u'icmpType',
                     u'ICMPV6_CODE': u'icmpv6Code',
                     u'ICMPV6_TYPE': u'icmpv6Type',
                     u'INNER_VLAN_PCP': u'innerPriority',
                     u'INNER_VLAN_VID': u'innerVlanId',
                     u'IN_PHY_PORT': u'port',
                     u'IN_PORT': u'port',
                     u'IPV4_DST': u'ip',
                     u'IPV4_SRC': u'ip',
                     u'IPV6_DST': u'ip',
                     u'IPV6_EXTHDR': u'exthdrFlags',
                     u'IPV6_FLABEL': u'flowlabel',
                     u'IPV6_ND_SLL': u'mac',
                     u'IPV6_ND_TARGET': u'targetAddress',
                     u'IPV6_ND_TLL': u'mac',
                     u'IPV6_SRC': u'ip',
                     u'IP_DSCP': u'ipDscp',
                     u'IP_ECN': u'ipEcn',
                     u'IP_PROTO': u'protocol',
                     u'METADATA': u'metadata',
                     u'MPLS_LABEL': u'label',
                     u'OCH_SIGID': u'ochSignalId',
                     u'OCH_SIGTYPE': u'ochSignalType',
                     u'ODU_SIGTYPE': u'oduSignalType',
                     u'SCTP_DST': u'sctpPort',
                     u'SCTP_SRC': u'sctpPort',
                     u'SLOT_GRANULARITY': u'slotGranularity',
                     u'SPACING_MULIPLIER': u'spacingMultiplier',
                     u'TCP_DST': u'tcpPort',
                     u'TCP_SRC': u'tcpPort',
                     u'TUNNEL_ID': u'tunnelId',
                     u'UDP_DST': u'udpPort',
                     u'UDP_SRC': u'udpPort',
                     u'VLAN_PCP': u'priority',
                     u'VLAN_VID': u'vlanId'}

        # HACK: ONOS REST API requires subnet bitmask for some reason...
        if 'ipv4_src' in kwargs and '/' not in kwargs['ipv4_src']:
            kwargs['ipv4_src'] += "/32"
        if 'ipv4_dst' in kwargs and '/' not in kwargs['ipv4_dst']:
            kwargs['ipv4_dst'] += "/32"
        if 'ipv6_src' in kwargs and '/' not in kwargs['ipv6_src']:
            kwargs['ipv6_src'] += "/128"
        if 'ipv6_dst' in kwargs and '/' not in kwargs['ipv6_dst']:
            kwargs['ipv6_dst'] += "/128"

        matches = [{"type": k.upper(), key_names[k.upper()]: v} for k,v in kwargs.items()]
        return matches

    def get_actions(self, *args):
        actions = []
        for a in args:
            if isinstance(a, str) or len(a) == 1:
                raise NotImplementedError("ONOS REST API doesn't accept no-argument actions")
            if len(a) == 2:
                action_type = a[0]
                value = a[1]
                if action_type.startswith("set_"):
                    if "ip" in action_type:
                        _type = "L4MODIFICATION"
                        value_name = "ip"
                    elif "eth" in action_type:
                        _type = "L2MODIFICATION"
                        value_name = "mac"
                    elif "vlan" in action_type:
                        raise NotImplementedError("VLAN modifications not yet implemented")
                    elif "tcp" in action_type:
                        _type = "L4MODIFICATION"
                        value_name = "tcpPort"
                    elif "udp" in action_type:
                        _type = "L4MODIFICATION"
                        value_name = "udpPort"
                    else:
                        raise ValueError("Unrecognized 'set' action %s" % a)
                    new_action = {"type": _type, "subtype": action_type[4:].upper(), value_name: value}
                elif action_type == "output":
                    new_action = {"type": "OUTPUT", "port": str(value)}
                elif action_type == "table":
                    new_action = {"type": "TABLE", "tableId": int(value)}
                elif action_type == "group":
                    new_action = {"type": "GROUP", "groupId": str(value)}
                else:
                    raise ValueError("Unrecognized or unimplemented action %s" % a)
                actions.append(new_action)
            else:
                raise NotImplementedError("Multi-argument actions not yet implemented")

        return actions

    def get_bucket(self, actions, weight=None, watch_group=None, watch_port=None):
        bucket = {'treatment': {'instructions': actions}}
        if weight is not None:
            bucket['weight'] = weight
        if watch_port is not None:
            bucket['watchPort'] = str(watch_port)
        if watch_group is not None:
            bucket['watchGroup'] = str(watch_group)
        return bucket

    def get_group_flow_rule(self, switch, buckets, group_id='1', group_type='ALL', **kwargs):
        """Builds a group flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param buckets - list of buckets where each bucket is formatted as returned from get_bucket(...)
        @:param type - type of group (all, indirect, select, fast_failover); defaults to 'all'

        @:return rule - dict representing the flow rule that can be installed on the switch"""

        # These are required fields
        rule = {"type": group_type,
                "deviceId": switch,
                "appCookie": "SdnTopology",
                "groupId": group_id,
                "buckets": buckets}
        return rule

    def __get_flow_rule(self, switch, **kwargs):
        """Helper function to assemble fields of a flow common between flow entry types."""
        rule = {"deviceId": switch, "isPermanent": True,
                "priority": 10}  # priority is required so we set a default here
        rule.update(kwargs)
        return rule

