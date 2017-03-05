__author__ = 'Kyle Benson (kebenson@uci.edu)'
"""Somewhat generic and helpful algorithms that other networkx users
might benefit from."""

import networkx as nx


def draw_overlaid_graphs(original, new_graphs, print_text=False):
    """Draws the new_graphs as graphs overlaid on the original topology"""
    import matplotlib.pyplot as plt

    layout = nx.spring_layout(original)
    nx.draw_networkx(original, pos=layout)
    # want to overlay edges in different colors and progressively thinner
    # so that we can see what edges are in a tree
    line_colors = 'rbgycm'
    line_width = 2.0 ** (min(len(new_graphs), len(line_colors)) - 1)  # use this for visualising on screen
    # line_width = 3.0 ** (min(len(new_graphs), len(line_colors)) - 1)  # use this for generating diagrams
    for i, g in enumerate(new_graphs):
        if print_text:
            print nx.info(g)
            print "edges:", list(g.edges())
        nx.draw_networkx(g, pos=layout, edge_color=line_colors[i % len(line_colors)], width=line_width)
        # advance to next line width
        line_width /= 1.7  # use this for visualising on screen
        # line_width /= 2.0  # use this for generating diagrams
    plt.show()


def node_split_in_out(G, nbunch=None, in_place=False, rename_func=None):
    """Return a DiGraph copy of the graph G with each node in nbunch
    (default is all nodes) split into two nodes n and n': the first
    has all the incoming edges of n and the second all the outgoing
    edges of n with edge (n,n') added too.

    NOTE: if G is a MultiDiGraph, any multiple outgoing edges with keys other
    than 0 will not be properly split!

    :type G: nx.DiGraph
    :param nbunch - nodes to split (iterable/iterator)
    :param in_place - if True, modifies graph directly rather than making a copy
    :param rename_func - function for renaming n' (default assumes\
    nodes are strings and makes n->n'
    :type return: nx.DiGraph
    """

    if nbunch is None:
        nbunch = G.nodes()
    if rename_func is None:
        def rename_func(n):
            return n + "'"

    if in_place:
        g2 = G
    else:
        g2 = G.to_directed()  # copy if directed, convert if not

    for n in nbunch:
        n2 = rename_func(n)
        g2.add_node(n2)

        # NOTE: need to include any attributes in edges!
        g2.add_edges_from((n2, neighbor, g2[n][neighbor][0]\
            if g2.is_multigraph() else g2[n][neighbor]) for neighbor in g2.successors(n))
        g2.remove_edges_from([(n, neighbor) for neighbor in g2.successors(n)])
        # Try including any node attributes (e.g. weight) in this edge
        g2.add_edge(n, n2, **G.node[n])

    # print '\n'.join([str(x) for x in g2.edges(data=True)])
    return g2

def get_redundant_paths(G, source, target, k=2, weight='weight'):
    """Gets k (possibly shortest) redundant paths with minimal component overlap.
    Current version based on Zheng et al 2010 paper entitled
    'Minimum-Cost Multiple Paths Subject to Minimum Link and Node Sharing in a Network'.
    The basic idea is to use network flow on a modified graph where each edge can
    handle one flow at regular cost but any others have greatly increased cost.
    This implementation assumes we only care about min-sum costs of edges then nodes
    for the constraints."""

    # 3-step algorithm: build transformed graph(s), find min-cost flow, compute paths
    if source == target:
        raise ValueError("source and target cannot be the same!")

    # First, build transformed graph by splitting each non-s/t node into two:
    # one for incoming nodes and one for outgoing nodes
    # (reduces node sharing problem to edge sharing)
    # NOTE: we do this in-place and so must specify a list not a generator for nbunch
    g2 = nx.MultiDiGraph(G)
    nodes_to_split = [n for n in g2.nodes() if (n != source and n != target)]
    g2 = node_split_in_out(g2, nodes_to_split, True)

    # Second, split each edge into two: original with unit capacity and regular cost;
    # the other with k-1 capacity and greatly increased costs to penalize choosing
    # an edge more than once.

    # Choose m1 and m2, which form the aforementioned penalties
    # NOTE: we need max_weight > 1 so we just fudge it here.  The
    # original paper said to modify all weights, but we don't.
    max_weight = max(G[u][v].get(weight, 1) for u,v in G.edges())
    if max_weight <= 1:
        max_weight += 1 - max_weight + 0.001
    # m2 discourages selecting a link multiple times, so ensure we'll
    # select k other longest paths before selecting this link again
    # ENHANCE: could this actually be k-2 and V-2?  says 'larger than
    # total cost of any k s-t loop-free paths', which might mean we don't
    # need max_weight > 1 either if we aren't squaring m2
    m2 = k * G.number_of_nodes() * max_weight
    # m1 discourages selecting a node multiple times, so we ensure we'd
    # select every other node k times before this one again
    m1 = m2 * k * G.number_of_nodes() * 1.001

    # Each edge needs capacity and weight, which is different depending on the end-points:
    # u-v' link gets += M2 and v-v' link gets = M1
    for u,v in list(g2.edges()):
        is_vv_link = v == u + "'"

        this_weight = m1 if is_vv_link else (g2[u][v][0].get(weight, 1) + m2)
        g2[u][v][0]['capacity'] = 1
        # Try copying existing weight from node v to its v-v' link
        # NOTE: we need to have non-zero weight on these links or
        # the flow algorithm will needlessly assign flow to them
        if is_vv_link:
            g2[u][v][0][weight] = G.node[u].get(weight, 0.00000001)
        g2.add_edge(u, v, 'bar', capacity=k-1, weight=this_weight)

    # Now it's time to find the min-cost flow
    g2.node[source]['demand'] = -k
    g2.node[target]['demand'] = k
    cost, flow = nx.capacity_scaling(g2)

    # Finally, generate the paths based on the flow found.
    # First, we build a flow graph with 0-flow edges removed.
    # Include the weight if possible
    flow_edges = ((u, v, key, dict(flow=flow[u][v][key], weight=g2[u][v][key].get(weight, 0)))\
                  for u in flow.keys() for v in flow[u].keys()\
                  for key in flow[u][v].keys() if flow[u][v][key] > 0)
    flow_graph = nx.MultiDiGraph(flow_edges)
    # Then, we gather k paths from the flow graph, deleting 0-flow edges
    # to avoid traversing them again in future paths.
    paths = []
    for i in range(k):
        p = nx.shortest_path(flow_graph, source, target, weight='weight')
        for u,v in zip(p, p[1:]):
            # Prefer primary edges first, which cannot have >1 flow
            try:
                flow_graph.remove_edge(u, v, 0)
            except nx.NetworkXError as e:
                flow_graph[u][v]['bar']['flow'] -= 1
                if flow_graph[u][v]['bar']['flow'] <= 0:
                    flow_graph.remove_edge(u, v, 'bar')
        paths.append([v for v in p if not v.endswith("'")])

    # sanity check
    if flow_graph.number_of_edges() > 0:
        print "WARNING! flow_graph still has flow edges left!", list(flow_graph.edges(data=True))

    return paths


# Simple tests
if __name__ == '__main__':
    npaths = 4

    g = nx.complete_graph(4)
    g.add_edge(0, 4)
    g.add_edge(3, 5)
    # Need to relabel to strings since we assume nodes are strings
    nx.relabel_nodes(g, {i: str(i) for i in g.nodes()}, copy=False)

    paths = get_redundant_paths(g, '4', '5', npaths)
    print paths