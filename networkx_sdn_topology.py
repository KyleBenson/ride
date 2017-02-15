import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

import networkx as nx
from sdn_topology import SdnTopology


class NetworkxSdnTopology(SdnTopology):
    """Generates a networkx topology (undirected graph) from information
    read directly from a networkx graph exported as JSON.
    Supports various functions such as finding multicast spanning trees and
    counting number of flow rules that would be installed in a real setting.

    The inheritance hierarchy works like this: the base class implements
    most of the interesting algorithms by using various helper functions.
    The derived classes implement those helper functions in order to adapt
    a particular data model and API (e.g. SDN controller, generic graph, etc.)
    to the SdnTopology tool."""

    def __init__(self, filename='campus_topo.json'):
        """@:param filename - name of Networkx JSON-formatted topology file to read
         and initialize network topology from."""
        super(NetworkxSdnTopology, self).__init__()
        self.filename = filename
        self.build_topology(filename)

    def get_info(self):
        return nx.info(self.topo)

    def get_hosts(self):
        return [n for n in self.topo.node if self.is_host(n)]

    def get_servers(self):
        return [n for n in self.topo.node if self.is_server(n)]

    def get_links(self, building_switches=False):
        """Return all links, optionally excluding those within
        a building."""
        return [(n1, n2) for n1, n2 in self.topo.edges() if
                (building_switches or self.is_switch(n1, building_switches) or self.is_switch(n2, building_switches))]

    def get_switches(self, building_switches=False):
        """Returns all switches, optionally excluding those within
        a building other than the building router."""
        return [n for n in self.topo.node if self.is_switch(n, building_switches)]

    def is_server(self, node):
        return node.startswith('s')

    def is_switch(self, node, include_building_switches=True):
        """Returns true if the node is a switch; false if it is not
        or the node is a switch within a building other than the building router."""
        if include_building_switches:
            return node[0] in ('c', 'd', 'b', 'm', 'f', 'r')
        else:
            return node[0] in ('c', 'd', 'b', 'm')

    # Overridden methods

    def build_topology(self, filename=None):
        if filename is None:
            filename = self.filename
        self.load_from_file(filename)

    def is_host(self, node):
        """Returns True if the given node is a host, False if it is a switch."""
        return node.startswith('h')

    def get_ip_address(self, host):
        """Gets the IP address associated with the given host in the topology.
        Currently simply returns the host ID (number)."""
        ip = self.topo.node[host][1:]
        return ip

    def get_ports_for_nodes(self, n1, n2):
        """Returns a pair of port numbers (or IDs) corresponding with the link
        connecting the two specified nodes respectively.  More than one link
        connecting the nodes is undefined behavior."""

        raise NotImplementedError()

        # TODO: check whether the storage mechanism of networkx will ensure ordering of edges for this function
        # Because of the undirected graph model, we have to disambiguate
        # the directionality of the request in order to properly order
        # the return values.

if __name__ == '__main__':
    st = NetworkxSdnTopology()
    print nx.info(st.topo)