import math
import itertools
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
        self.server_nodes = None
        self.major_building_routers = None
        self.minor_building_routers = None
        self.distribution_routers = None
        self.hosts = None

    def generate(self):
        """Generates and returns the topology."""

        # Start with core as a complete graph
        self.topo = nx.Graph()
        self.topo.graph['name'] = "Campus Network Topology"
        for src, dst in itertools.combinations(range(self.core_size), 2):
            self.add_link("c%d" % src, "c%d" % dst)
        self.core_nodes = list(self.topo.nodes())

        # Server(s), e.g. data centers, are assumed to be located close to the core
        self.server_nodes = []
        for s in range(self.servers):
            server_name = "s%d" % s
            self.topo.add_node(server_name)
            self.server_nodes.append(server_name)
            for server_router in random.sample(self.core_nodes, self.links_per_server):
                self.add_link(server_router, server_name)

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
                self.add_link(node_name, dst)

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
                self.add_link(router_name, dst)

            # evenly balance out the extra minor buildings added,
            # keeping in mind that we may not have had enough minor
            # buildings to fully justify a whole distribution router
            # and so may only place the remaining minors
            mbs_to_add = min(self.minor_buildings_per_distribution_router, nminor_buildings - len(self.minor_building_routers))
            if extra_minors > 0:
                mbs_to_add += 1
                extra_minors -= 1

            for j in range(mbs_to_add):
                node_name = "m%d" % len(self.minor_building_routers)
                self.topo.add_node(node_name)
                self.add_link(node_name, router_name)
                self.minor_building_routers.append(node_name)

        # add in-building topologies and/or just hosts
        self.hosts = []
        for b in self.major_building_routers:
            self.create_building_topology(b)

        for b in self.minor_building_routers:
            self.create_building_topology(b, False, True)

        self.add_inter_building_links()

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
                self.add_link(building, floor_switch_name)
                for rack_switch in range(self.building_switches_per_floor):
                    rack_switch_name = "r%d-%s" % (rack_switch, floor_switch_name)
                    self.topo.add_node(rack_switch_name)
                    self.add_link(floor_switch_name, rack_switch_name)
                    for host in range(self.hosts_per_floor_switch):
                        host_name = "h%d-%s" % (host, rack_switch_name)
                        self.topo.add_node(host_name)
                        self.add_link(rack_switch_name, host_name)
                        self.hosts.append(host_name)
        else:
            if is_minor_building:
                nhosts = self.hosts_per_floor_switch
            else:
                nhosts = self.hosts_per_floor_switch * self.building_switches_per_floor * self.building_floors
            for host in range(nhosts):
                host_name = "h%d-%s" % (host, building)
                self.topo.add_node(host_name)
                self.add_link(building, host_name)
                self.hosts.append(host_name)

    def add_inter_building_links(self):
        """Adds links between buildings randomly for additional
        redundancy and topology diversity."""

        try:
            endpoints = random.sample(self.major_building_routers, self.inter_building_links * 2)
            endpoints = zip(endpoints[:len(endpoints)/2], endpoints[len(endpoints)/2:])
        except ValueError as e:
            print "NOTE: requested more inter_building_links" \
                  " than can be placed without repeating (major) buildings!"
            if self.inter_building_links > 400:
                print "Requested a lot of inter-building links.  This may take a while to generate all combinations without repeat..."
            endpoints = list(itertools.combinations_with_replacement(self.major_building_routers, 2))
            random.shuffle(endpoints)
            endpoints = endpoints[:self.inter_building_links]

        for src, dst in endpoints:
            self.add_link(src, dst)

    def add_link(self, src, dst):
        """Add a link between the src and dst. Sets various attributes."""
        latency = self.get_link_latency(src, dst)
        weight = self.get_link_weight(src, dst)
        # TODO: bandwidth
        self.topo.add_edge(src, dst, latency=latency, weight=weight)

    def get_link_weight(self, src, dst):
        """Assign a link weight, which represents cost of operation,
        as determined by the types of end-points:
          (building | distribution)-core or minor building:
                low as necessary part of routing to hosts
          inter-core: medium as fiber but carrying high load
          server: high as assumed resource constrained / expensive
          inter-building: highest as assumed for local traffic / emergency backup

        :param str src:
        :param str dst:
        """
        # server
        if src.startswith('s') or dst.startswith('s'):
            return 1.4
        # inter-building
        elif src.startswith('b') and dst.startswith('b'):
            return 1.8
        # inter-core
        elif src.startswith('c') and dst.startswith('c'):
            return 1.2
        # TODO: should support the intra-building link types: 0 weight?  hosts should be too then...
        # hosts, building-core, distribution-core, or minor building
        else:
            return 1.0

    def get_link_latency(self, src, dst):
        """Randomly generate a link latency within a range determined
        by the types of end-points:
          server: lowest as assumed directly connected
          inter-building: low as assumed close
          building-core: medium-low as fiber links but relatively close
          hosts: medium-low as close proximity but ethernet
          inter-core: medium as fiber links but far apart
          distribution-core: high as assumed older tech
          minor building: highest as assumed old tech

        :param str src:
        :param str dst:
        """
        # sort to make these if statements easier
        src, dst = tuple(sorted((src, dst)))
        # server
        if dst.startswith('s'):
            start = 1
            stop = 3
        # host
        elif src.startswith('h') or dst.startswith('h'):
            # TODO: should support the intra-building link types here
            start = 5
            stop = 10
        elif src.startswith('b'):
            # inter-building
            if dst.startswith('b'):
                start = 2
                stop = 8
            # building-core
            elif dst.startswith('c'):
                start = 7
                stop = 15
        elif src.startswith('c'):
            # inter-core
            if dst.startswith('c'):
                start = 10
                stop = 20
            # distribution-core
            elif dst.startswith('d'):
                start = 20
                stop = 30
        # minor building
        elif dst.startswith('m') and src.startswith('d'):
            start = 25
            stop = 50
        else:
            raise TypeError("Didn't recognize the src/dst router types: %s, %s" % (src, dst))
        return random.uniform(start, stop)

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
            # shell layout places nodes as a series of concentric circles
            positions = nx.shell_layout(self.topo, [self.core_nodes,
                                                    # sort the building routers by degree in attempt to get ones connected to each other next to each other
                                                    sorted(self.major_building_routers, key=lambda n: nx.degree(self.topo, n)) + self.distribution_routers + self.server_nodes,
                                                    self.hosts + self.minor_building_routers])
            # then do a spring layout, keeping the inner nodes fixed in positions
            positions = nx.spring_layout(self.topo, pos=positions, fixed=self.core_nodes + self.server_nodes + self.major_building_routers + self.distribution_routers)
            nx.draw(self.topo, node_color=node_colors, pos=positions)
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
                                    add_building_topology=False, inter_building_links=3)
    else:
        # build multiple topologies and save each of them
        # this one will be the default topo for testing
        # t = CampusTopologyGenerator(nbuildings=8, hosts_per_floor_switch=2,
        #                             building_switches_per_floor=1, building_floors=2,
        #                             add_building_topology=False, inter_building_links=2)
        # t.generate()
        # t.write()

        # iterate over multiple options of form (nbuildings, nhosts, n-inter-building-links)
        topologies_to_build = (
            # (20, 8, 3), (40, 8, 5), (50, 8, 6), (80, 8, 8),  # smaller topologies
            # (200, 20, 20),  # main large topology
            (10, 4, 2),
            # (200, 20, 40), (200, 20, 80),
            # (200, 20, 10), (200, 20, 0), (200, 20, 60), # vary ibl on main topology
            # (200, 20, 200), (200, 20, 400), (200, 20, 800), # vary ibl on main topology, with repeats and larger #s
            # (200, 8, 20), (80, 16, 8), (200, 40, 20),  # keeping constant host:nbuilds ratio
            # (400, 80, 400),  # did this with core_size=8
        )
        for nb, nh, nibl in topologies_to_build:
            print "Generating topo with %d buildings, %d hosts, and %d inter-building links" % (nb, nh, nibl)
            t = CampusTopologyGenerator(nbuildings=nb, hosts_per_floor_switch=nh,
                                        building_switches_per_floor=1, building_floors=1,
                                        add_building_topology=False, inter_building_links=nibl,
                                        # core_size=16, links_per_building=4, links_per_server=8,
                                        )
            t.generate()
            t.write('campus_topo_%db-%dh-%dibl.json' % (nb, nh, nibl))

    if test_run:
        g = t.get()
        print nx.info(g)
        t.draw()
        t.write()
