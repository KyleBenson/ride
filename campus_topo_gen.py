import math
import networkx as nx
from networkx.readwrite import json_graph
import json
import random

class CampusTopologyGenerator(object):
    """Generates a networkx graph representing a campus network topology.
    Assumes a core of fully-connected routers, major building routers that
    connect to two core routers, and minor building switches that connect to
    distribution routers that connect to two differentc core routers.

    Buildings are optionally fitted with a depth 3 tree topology representing
    the switches and jacks in the building.

    See constructor for default values

    Future versions may include additional redundant links, attributes such as
    latency or bandwidth, and shared risk groups."""

    def __init__(self, core_size=4, servers=1, links_per_server=2,
                 percent_minor_buildings=0.15, # 10-15%
                 minor_buildings_per_distribution_router=8, # 7-8
                 links_per_distribution_router=2, links_per_building=2, nbuildings=200,
                 add_building_topology=True,  # building topo is a tree, so no redundancy or clever routing to be done
                 building_floors=5,  # estimate for larger buildings, should probably make a distribution
                 building_switches_per_floor=5,  # 4-6
                 hosts_per_floor_switch=10, # assuming just potentially relevant IoT devices.  should be much more...
                 inter_building_links=6, # 5-7
                 # nsrlg_groups=4 # plus two sub-regions per group?
                 ):
        super(CampusTopologyGenerator, self).__init__()

        self.core_size = core_size
        self.servers = servers
        self.links_per_server = links_per_server
        self.percent_minor_buildings = percent_minor_buildings
        self.minor_buildings_per_distribution_router = minor_buildings_per_distribution_router
        self.links_per_distribution_router = links_per_distribution_router
        self.links_per_building = links_per_building
        self.nbuildings = nbuildings
        self.add_building_topology = add_building_topology
        self.building_floors = building_floors
        self.building_switches_per_floor = building_switches_per_floor
        self.hosts_per_floor_switch = hosts_per_floor_switch
        self.inter_building_links = inter_building_links

        self.topo = None
        self.core_nodes = None
        self.major_building_routers = None
        self.minor_building_routers = None
        self.distribution_routers = None
        self.hosts = None

    def generate(self):
        """Generates and returns the topology."""

        # Start with core
        self.topo = nx.complete_graph(self.core_size)
        self.topo.graph['name'] = "Campus Network Topology"
        nx.relabel_nodes(self.topo, {i: "c%d" % i for i in range(self.core_size)}, copy=False)
        self.core_nodes = list(nx.nodes(self.topo))

        # Server(s), e.g. data centers, are assumed to be located close to the core
        for s in range(self.servers):
            server_name = "s%d" % s
            self.topo.add_node(server_name)
            for server_router in random.sample(self.core_nodes, self.links_per_server):
                self.topo.add_edge(server_router, server_name)

        # Add buildings
        nminor_buildings = int(math.ceil(self.nbuildings * self.percent_minor_buildings))
        nmajor_buildings = self.nbuildings - nminor_buildings
        self.major_building_routers = []
        self.minor_building_routers = []
        self.distribution_routers = []

        print "adding %d major and %d minor buildings" % (nmajor_buildings, nminor_buildings)

        for i in range(nmajor_buildings):
            node_name = "b%d" % i
            self.topo.add_node(node_name)
            self.major_building_routers.append(node_name)
            # Add links to core
            for dst in random.sample(self.core_nodes, self.links_per_building):
                self.topo.add_edge(node_name, dst)

        # Add distribution routers that support minor buildings (rounding down);
        # then connect them to the minor buildings.  If we have too few minor
        # buildings to justify a distribution router, should place one anyway.
        ndist_routers = nminor_buildings / self.minor_buildings_per_distribution_router
        extra_minors = nminor_buildings % self.minor_buildings_per_distribution_router
        if ndist_routers < 1 and nminor_buildings > 0:
            ndist_routers = 1
            extra_minors = 0
        print "adding %d dist routers" % ndist_routers
        for i in range(ndist_routers):
            router_name = "d%d" % i
            self.topo.add_node(router_name)
            self.distribution_routers.append(router_name)
            for dst in random.sample(self.core_nodes, self.links_per_distribution_router):
                self.topo.add_edge(router_name, dst)

            # evenly balance out the extra minor buildings added,
            # keeping in mind that we may not have had enough minor
            # buildings to fully justify a whole distribution router
            # and so may only place the remaining minors
            mbs_to_add = min(self.minor_buildings_per_distribution_router, nminor_buildings - len(self.minor_building_routers))
            if extra_minors > 0:
                mbs_to_add += 1
                extra_minors -= 1

            for j in range(mbs_to_add):
                node_name = "m%d" % j
                self.topo.add_node(node_name)
                self.topo.add_edge(node_name, router_name)
                self.minor_building_routers.append(node_name)

        # add in-building topologies and/or just hosts
        self.hosts = []
        for b in self.major_building_routers:
            self.create_building_topology(b)

        for b in self.minor_building_routers:
            self.create_building_topology(b, False, True)

        # TODO: add inter-building links

        print "Added %d hosts" % len(self.hosts)

        return self.topo

    def create_building_topology(self, building, add_internal_switches=None, is_minor_building=False):
        """Creates a depth 3 tree topology of switches as a building topology
        if self.add_building_topology is True, else it simply creates a bunch
        of hosts as leaves in this building.  Note that this treatment is also
        dependent on whether this is a minor building (assumes 1 floor and
        1 switch that IS the already-placed router)."""

        if add_internal_switches is None:
            add_internal_switches = self.add_building_topology

        if add_internal_switches and not is_minor_building:
            # CONSIDER: give each host a unique host #
            for floor in range(self.building_floors):
                floor_switch_name = "f%d-%s" % (floor, building)
                self.topo.add_node(floor_switch_name)
                self.topo.add_edge(building, floor_switch_name)
                for rack_switch in range(self.building_switches_per_floor):
                    rack_switch_name = "r%d-%s" % (rack_switch, floor_switch_name)
                    self.topo.add_node(rack_switch_name)
                    self.topo.add_edge(floor_switch_name, rack_switch_name)
                    for host in range(self.hosts_per_floor_switch):
                        host_name = "h%d-%s" % (host, rack_switch_name)
                        self.topo.add_node(host_name)
                        self.topo.add_edge(rack_switch_name, host_name)
                        self.hosts.append(host_name)
        else:
            if is_minor_building:
                nhosts = self.hosts_per_floor_switch
            else:
                nhosts = self.hosts_per_floor_switch * self.building_switches_per_floor * self.building_floors
            for host in range(nhosts):
                host_name = "h%d-%s" % (host, building)
                self.topo.add_node(host_name)
                self.topo.add_edge(building, host_name)
                self.hosts.append(host_name)


    def get(self):
        if self.topo is not None:
            return self.topo
        else:
            return self.generate()

    def draw(self):
        """Draw the graph using matplotlib in a color-coordinated manner."""
        try:
            import matplotlib.pyplot as plt
            print 'Node colors: red=core, blue=major-building, green=distribution, yellow=minor-building, cyan=server, magenta=host, black=floor-switch, white=rack-switch'
            # TODO: ignore building internals?
            colormap = {'c': 'r', 'b': 'b', 'd': 'g', 'm': 'y', 's': 'c', 'h': 'm', 'f': 'k', 'r': 'w'}
            node_colors = [colormap[node[0]] for node in self.topo.nodes()]
            nx.draw(self.topo, node_color=node_colors)
            plt.show()
        except ImportError:
            print "ERROR: couldn't draw graph as matplotlib.pyplot couldn't be imported!"

    def write(self, filename='campus_topo.json'):
        """Writes the generated network topology to JSON file."""
        data = json_graph.node_link_data(self.topo)
        with open(filename, "w") as f:
            json.dump(data, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    test_run = False

    if test_run:
        # build smaller topology for visualizing
        t = CampusTopologyGenerator(nbuildings=8, hosts_per_floor_switch=2,
                                    building_switches_per_floor=1, building_floors=2,
                                    add_building_topology=False)
    else:
        t = CampusTopologyGenerator(nbuildings=80, hosts_per_floor_switch=4,
                                    building_switches_per_floor=1, building_floors=2,
                                    add_building_topology=False)

    g = t.get()
    print nx.info(g)
    if test_run:
        t.draw()
    t.write()
