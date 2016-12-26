import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO)

from floodlight_api import RestApi
import json
import networkx as nx


class SdnTopology(object):
    """Generates a networkx topology from information gleaned from an SDN Controller.
    Supports various functions such as finding multicast spanning trees and
    installing flow rules."""

    def __init__(self, ip='localhost', port='8080'):
        super(SdnTopology, self).__init__()

        self.rest = RestApi(ip, port)
        self.topo = nx.Graph()

        self.build_topology()

    def build_topology(self):

        cmd = 'switches'
        args = []

        path = self.rest.lookup_path(cmd, args)
        switches = json.loads(self.rest.get(path))

        log.debug(json.dumps(switches, sort_keys=True, indent=4))

        for s in switches:
            self.topo.add_node(s['switchDPID'])

        log.debug(self.topo.nodes())


        cmd = 'link'
        path = self.rest.lookup_path(cmd, args)
        links = json.loads(self.rest.get(path))

        log.debug(json.dumps(links, sort_keys=True, indent=4))

        for link in links:
            self.topo.add_edge(link['src-switch'], link['dst-switch'], latency=link['latency'],
                               port1={'dpid': link['src-switch'], 'port_num': link['src-port']},
                               port2={'dpid': link['dst-switch'], 'port_num': link['dst-port']})

        log.debug(self.topo.edges(data=True))

        cmd = 'hosts'
        path = self.rest.lookup_path(cmd, args)
        hosts = json.loads(self.rest.get(path))

        log.debug(json.dumps(hosts, sort_keys=True, indent=4))

        for host in hosts['devices']:
            # assume hosts only have a single IP address/interface/MAC address/attachment point,
            # but potentially multiple VLANs
            try:
                ipv4 = host['ipv4'][0]
            except IndexError:
                print "Skipping host with no IP address: %s" % host
                continue

            self.topo.add_node(ipv4, mac=host['mac'][0], vlan=host['vlan'])
            # TODO: generalize the creation of an edges' attributes? maybe similar for adding a host?
            # NOTE: we are using the ipv4 address in the DPID field of the host, this may be wrong...
            switch = host['attachmentPoint'][0]
            self.topo.add_edge(ipv4, switch['switch'], latency=0,
                               port1={'dpid': ipv4, 'port_num': 0},
                               port2={'dpid': switch['switch'], 'port_num': switch['port']})

        log.info("Final %d nodes: %s" % (len(self.topo.nodes()), self.topo.nodes()))
        log.info("Final %d edges: %s" % (len(self.topo.edges()), self.topo.edges()))


if __name__ == '__main__':
    st = SdnTopology()