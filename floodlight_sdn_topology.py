import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

from floodlight_api import FloodlightRestApi
from sdn_topology import SdnTopology
import json
import networkx as nx

# Otherwise, identify by ipv4
IDENTIFY_HOSTS_BY_MAC = True


class FloodlightSdnTopology(SdnTopology):
    """Generates a networkx topology from information gleaned from an SDN Controller.
    Supports various functions such as finding multicast spanning trees and
    installing flow rules."""

    def __init__(self, ip='localhost', port='8080'):
        super(FloodlightSdnTopology, self).__init__()

        self.rest_api = FloodlightRestApi(ip, port)
        self.unique_counter = 0  # used for e.g. flow entry names

        self.build_topology()

    # Topology management helper functions

    def add_link(self, link):
        """Adds the given link, in its raw input format, to the topology."""
        self.topo.add_edge(link['src-switch'], link['dst-switch'], latency=link['latency'],
                           port1={'dpid': link['src-switch'], 'port_num': link['src-port']},
                           port2={'dpid': link['dst-switch'], 'port_num': link['dst-port']})

    def add_switch(self, switch):
        """Adds the given switch, in its raw input format, to the topology."""
        self.topo.add_node(switch['switchDPID'])

    def add_host(self, host):
        """Adds the given host, in its raw input format, to the topology."""
        # assume hosts only have a single IP address/interface/MAC address/attachment point,
        # but potentially multiple VLANs
        mac = ipv4 = None

        try:
            switch = host['attachmentPoint'][0]
        except IndexError:
            log.debug("Skipping host with no attachmentPoint: %s" % host)
            return

        try:
            mac = host['mac'][0]
            ipv4 = host['ipv4'][0]
        except IndexError:
            if mac is None:
                log.debug("Skipping host with no MAC or IPv4 addresses: %s" % host)
                return

        if IDENTIFY_HOSTS_BY_MAC:
            hostid = mac
        else:
            hostid = ipv4

        # TODO: turn the last kwarg into 'ip'?
        self.topo.add_node(hostid, mac=mac, vlan=host['vlan'], ip=ipv4)
        self.topo.add_edge(hostid, switch['switch'], latency=0,
                           port1={'dpid': hostid, 'port_num': 0},
                           # for some reason, REST API gives us port# as a str
                           port2={'dpid': switch['switch'], 'port_num': int(switch['port'])})

    def is_host(self, node):
        """Returns True if the given node is a host, False if it is a switch.
        Does NOT determine if the node is in the topology."""

        # Switches have more bits in 'MAC address' and
        # are never identified by IP address
        if node.count(':') == 7 and node.count('.') == 0:
            return False
        else:
            return True

    # Flow rule helper functions

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

    def get_bucket(self, actions, weight=None, watch_group=None, watch_port=None):
        """Formats a dict-like object to use as a bucket within get_group_flow_rule.

        @:param actions - actions to perform

        @:return bucket - dict representing the bucket with all necessary fields filled
        """

        bucket = {'bucket_actions': actions,
                  # HACK: this is necessary due to a bug in Floodlight that
                  # sets the value to "any" when unspecified, causing an error
                  "bucket_watch_group": ("any" if watch_group is None else watch_group)
                  }
        if weight is not None:
            bucket['bucket_weight'] = weight
        if watch_port is not None:
            bucket['bucket_watch_port'] = watch_port
        return bucket

    def get_group_flow_rule(self, switch, buckets, group_id='1', group_type='all', **kwargs):
        """Builds a group flow rule that can be installed on the corresponding switch via the RestApi.

        @:param switch - the DPID of the switch this flow rule corresponds with
        @:param buckets - list of buckets where each bucket is formatted as returned from get_bucket(...)
        @:param type - type of group (all, indirect, select, fast_failover); defaults to 'all'
        @:param **kwargs - all remaining kwargs are added to the flow rule dict using rule.update()

        @:return rule - dict representing the flow rule that can be installed on the switch"""

        rule = self.__get_flow_rule(switch, **kwargs)
        # Floodlight REST API uses a bucket_id field to prioritize the buckets,
        # but our API keeps them as an ordered list.  Hence, we need to add
        # this field when building the flow rule.
        for bucket_id,b in enumerate(buckets):
            b['bucket_id'] = bucket_id+1
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

    def get_matches(self, **kwargs):
        # ensure everything is formatted correctly, including types, for the picky REST API
        if 'in_port' in kwargs:
            kwargs['in_port'] = str(kwargs['in_port'])
        return kwargs

    def get_actions(self, *args):
        # Floodlight formats them as e.g. "strip_vlan,set_field=ipv4_dst->10.0.0.1,output=1"
        actions = []
        for a in args:
            if isinstance(a, str):
                actions.append(a)
            elif len(a) == 1:
                actions.append(a[0])
            elif a[0].startswith("set_") and len(a) == 2:
                # Assuming we're using OpenFlow 1.2+, we need to use the 'set_field' action
                actions.append("set_field=%s->%s" % (a[0][4:], a[1]))
            elif len(a) == 2:
                actions.append("%s=%s" % a)
            elif len(a) > 2:
                raise NotImplementedError("Actions with more than one argument not supported!")
            else:
                raise ValueError("Unsupported action %s" % a)

        return ",".join(actions)