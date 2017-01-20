import math
import networkx as nx
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

    def __init__(self, core_size=4, percent_minor_buildings=0.1, minor_buildings_per_distribution_router=10,
                 links_per_distribution_router=2, links_per_building=2, nbuildings=400,
                 add_building_topology=False, building_floors=5, building_switches_per_floor=5, hosts_per_floor_switch=):
        super(CampusTopologyGenerator, self).__init__()

        self.core_size = core_size
        self.percent_minor_buildings = percent_minor_buildings
        self.minor_buildings_per_distribution_router = minor_buildings_per_distribution_router
        self.links_per_distribution_router = links_per_distribution_router
        self.links_per_building = links_per_building
        self.nbuildings = nbuildings
        self.add_building_topology = add_building_topology
        self.building_floors = building_floors
        self.building_switches_per_floor = building_switches_per_floor

        self.topo = None
        self.core_nodes = None
        self.major_building_routers = None
        self.minor_building_routers = None
        self.distribution_routers = None

    def generate(self):
        """Generates and returns the topology."""
        # Start with core
        self.topo = nx.complete_graph(self.core_size)
        nx.relabel_nodes(self.topo, {i: "c%d" % i for i in range(self.core_size)}, copy=False)
        self.core_nodes = list(nx.nodes(self.topo))

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

        # Optionally add in-building topologies
        if self.add_building_topology:
            raise NotImplementedError("Currently don't support in-building topologies")

        return self.topo

    def get(self):
        if self.topo is not None:
            return self.topo
        else:
            return self.generate()

    def draw(self):
        """Draw the graph using matplotlib in a color-coordinated manner."""
        try:
            import matplotlib.pyplot as plt
            print 'Node colors: red=core, blue=major-building, green=distribution, yellow=minor-building'
            colormap = {'c': 'r', 'b': 'b', 'd': 'g', 'm': 'y'}
            node_colors = [colormap[node[0]] for node in g.nodes()]
            nx.draw(g, node_color=node_colors)
            plt.show()
        except ImportError:
            print "ERROR: couldn't draw graph as matplotlib.pyplot couldn't be imported!"


if __name__ == '__main__':
    t = CampusTopologyGenerator(nbuildings=8)
    g = t.get()
    print nx.info(g)
    g.draw()
