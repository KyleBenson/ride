import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

import networkx as nx


class NetworkTopology(object):
    """Uses networkx graph model and various algorithms to perform
    various networking-related computations such as paths, multicast
    trees, and redundant (reliable) variations of them.  This is where
    you should implement generic graph algorithms for use in the
    other SdnTopology classes."""

    def __init__(self, topo=None):
        """
        :type topo: nx.Graph
        """
        super(NetworkTopology, self).__init__()
        if topo is None:
            self.topo = nx.Graph()
        else:
            self.topo = topo

    def get_redundant_multicast_trees(self, source, destinations, k=2, algorithm='networkx'):
        """Builds k redundant multicast trees: trees should not share any edges
        unless necessary.  Supports various algorithms, several of which may not
        work for k>2."""

        if algorithm == 'networkx':
            """Default algorithm implemented by networkx that uses sum of
            shortest paths 2*D approximation.  Currently not available in
            latest release of networkx, so see README if this import doesn't work."""

            try:
                from networkx.algorithms.approximation import steiner_tree
            except ImportError:
                raise NotImplementedError("Steiner Tree algorithm not found!  See README")

            # we don't care about directionality of the mcast tree here,
            # so we can treat the source as yet another destination
            destinations.append(source)

            # Skip over graph modifications if we only want one tree
            if k == 1:
                # TODO: verify this keeps attributes intact
                return [steiner_tree(self.topo, destinations)]

            # Naive heuristic: generate a multicast tree, increase the
            # weights on the edges to discourage them, generate another...
            # So we need to add a temporary attribute to the edges for
            # the heuristic to use or else we'd overwrite the weights.
            # TODO: generalize this residual graph approach?

            for u,v in self.topo.edges():
                self.topo[u][v]['_temp_mcast_weight'] = self.topo[u][v].get('weight', 1.0)
            # Add the max weight of all edges to prevent an edge from being chosen next round
            max_weight = max((e[2]['_temp_mcast_weight'] for e in self.topo.edges(data=True)))

            trees = []
            for i in range(k):
                new_tree = steiner_tree(self.topo, destinations, weight='_temp_mcast_weight')
                for u,v in new_tree.edges():
                    self.topo[u][v]['_temp_mcast_weight'] += max_weight
                trees.append(new_tree)

            for u,v in self.topo.edges():
                del self.topo[u][v]['_temp_mcast_weight']

            return trees

        elif algorithm == 'ilp':
            """Our (UCI-DSM group) proposed ILP-based heuristic."""
            from redundant_multicast_algorithms import ilp_redundant_multicast
            return ilp_redundant_multicast(self.topo, source, destinations, k)

        else:
            raise ValueError("Unkown multicast tree generation algorithm %s" % algorithm)


    def get_multicast_tree(self, source, destinations, algorithm='networkx'):
        """Uses networkx algorithms to build a multicast tree for the given source node and
        destinations (an iterable).  Can be used to build and install flow rules.
        Current implementation simply calls to get_redundant_multicast_trees(k=1)
        Default algorithm uses the metric closure-based approximation of a steiner tree."""

        return self.get_redundant_multicast_trees(source, destinations, 1, algorithm)[0]

    # Path generation procedures

    def get_redundant_paths(self, source, destination, k=2):
        """Gets k (possibly shortest) redundant paths with minimal component overlap"""
        # TODO: if possible? else get maximally-redundant
        raise NotImplementedError

    def get_path(self, source, destination):
        """Gets shortest path by weight attribute between the nodes.
        @:return a sequence of nodes representing the shortest path"""

        return nx.shortest_path(self.topo, source=source, target=destination)

    def draw_multicast_trees(self, trees):
        # draw the results of the trees on top of the original network
        import matplotlib.pyplot as plt

        layout = nx.spring_layout(self.topo)
        nx.draw_networkx(self.topo, pos=layout)
        # want to overlay edges in different colors and progressively thinner
        # so that we can see what edges are in a tree
        line_colors = 'rbgycm'
        line_width = 2.0 ** (ntrees - 1)
        for t in trees:
            # print nx.info(m)
            # print "edges:", list(m.edges())
            nx.draw_networkx(t, pos=layout, edge_color=line_colors[0], width=line_width)
            # advance to next line color and width
            line_colors = line_colors[1:]
            line_width /= 1.5
        plt.show()


# Run various tests
if __name__ == '__main__':
    # algorithm = 'ilp'
    algorithm = 'networkx'
    ntrees = 4

    g = nx.complete_graph(4)
    g.add_edge(0, 4)
    g.add_edge(3, 5)
    # Need to relabel to strings since we assume nodes are strings
    nx.relabel_nodes(g, {i: str(i) for i in g.nodes()}, copy=False)
    net = NetworkTopology(g)

    dest = ["2", "5"]
    # dest = ["5"]
    M = net.get_redundant_multicast_trees("4", dest, ntrees, algorithm)

    net.draw_multicast_trees(M)

