__author__ = 'Kyle Benson (kebenson@uci.edu)'
"""Somewhat generic and helpful algorithms that other networkx users
might benefit from."""

import networkx as nx
import logging
log = logging.getLogger(__name__)


def draw_overlaid_graphs(original, new_graphs, print_text=False):
    """
    Draws the new_graphs as graphs overlaid on the original topology
    :type new_graphs: list[nx.Graph]
    """
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
            log.info(nx.info(g))
            log.info("edges:", list(g.edges()))
        nx.draw_networkx(g, pos=layout, edge_color=line_colors[i % len(line_colors)], width=line_width)
        # advance to next line width
        line_width /= 1.7  # use this for visualising on screen
        # line_width /= 2.0  # use this for generating diagrams
    plt.show()


def draw_paths(G, paths):
    """
    Draws the specified paths on the graph G by building a graph for each path and passing to draw_overlaid_graphs
    :type G: nx.Graph
    :type paths: collections.Iterable
    """

    GP = [nx.Graph(((u, v) for u, v in get_edges_for_path(p))) for p in paths]
    draw_overlaid_graphs(G, GP)


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
    for the constraints.
    WARNING: nodes are expected to be strings since we split them up and relabel them in a temp graph!"""

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
        for u,v in get_edges_for_path(p):
            # Prefer primary edges first, which cannot have >1 flow
            try:
                flow_graph.remove_edge(u, v, 0)
            except nx.NetworkXError as e:
                flow_graph[u][v]['bar']['flow'] -= 1
                if flow_graph[u][v]['bar']['flow'] <= 0:
                    flow_graph.remove_edge(u, v, 'bar')
        paths.append([v for v in p if not v.endswith("'")])

    # sanity check
    if __debug__:
        if flow_graph.number_of_edges() > 0:
            log.debug("flow_graph still has flow edges left! This might be indicative of a problem, but it seems to "
                      "happen somewhat regularly with our tree-like campus topologies... Remaining flows: %s"
                      % list(flow_graph.edges(data=True)))

    return paths


def get_multi_source_disjoint_paths(G, sources, target, weight='weight'):
    """Gets len(sources) (possibly shortest) maximally-disjoint paths (minimal component overlap) from
    multiple sources to one target destination.  This just calls get_redundant_paths() with a modified graph
    in which a new 'virtual node' is added with edges to each source.  Because the paths are maximally-disjoint and
    k=len(sources), each of these edges *should* be used and so each source *should* have a path to target.

    Based on our implementation of get_redundant_paths(), however, it's possible that some paths may not include a
     proper source due to the only remaining path to the source incurring such a high penalty that it's cheaper to
    re-use one of the 'virtual links'.  In such a case, we throw out the useless path (it's links probably weren't
    of interest to the other sources since they were so cheap still) and simply choose the normal shortest path for any
    sources not yet accounted for.  This path won't have disjointness guarantees, but we probably didn't have a better
    option for it anyway...  Could enhance this by actually doing the shortest path (from true source) on the the
    flow graph to get a cheaper path?

    :type G: nx.Graph
    :type sources: list|tuple|set
    :param target: target destination node in G
    :param weight: string specifying path length
    :raises nx.NetworkxError: if something goes wrong
    """

    G2 = nx.Graph(G)
    virt_node = '__multi_source_virt_node__'
    for s in sources:
        # make sure we specify a 0 weight of the proper attribute
        G2.add_edge(virt_node, s, **{weight: 0})

    paths = get_redundant_paths(G2, virt_node, target, k=len(sources), weight=weight)

    # Now we ensure that all sources are accounted for and filter out paths that don't start (index 1) with one of
    # our sources.
    sources = set(sources)
    sources_covered = set()
    final_paths = []
    for p in paths:
        # We might have multiple paths with same legitimate source, in which case we need to choose which one to keep:
        # ENHANCE: perhaps choose the one with the cheapest flow value based on having the other algorithm return the flows?
        # XXX: for now, we're just going to take the first one arbitrarily...
        src_node = p[1]
        if src_node in sources and src_node not in sources_covered:
            sources_covered.add(src_node)
            # chop off virt_node so we're using actual paths
            true_path = p[1:]
            final_paths.append(true_path)
            assert path_exists(G, true_path), "not a valid path! %s" % true_path
        assert p[-1] == target, "path contained incorrect target: expected %s but got %s" % (target, p[-1])

    # XXX: Finally, we need to add a regular shortest path for those that aren't
    # ENHANCE: could try to select a slightly more disjoint path (see above about using returned flow values)
    for leftover in sources - sources_covered:
        log.debug("assigning regular shortest path for node %s as it was missing from our diverse paths to target %s" % (leftover, target))
        final_paths.append(nx.shortest_path(G, source=leftover, target=target, weight=weight))
        assert final_paths[-1][0] == leftover
        assert final_paths[-1][0] != target

    assert len(final_paths) == len(sources)  # simple sanity check to ensure we're returning the right # paths!
    return final_paths


def path_exists(G, p):
    """
    Returns true if p is a valid path in G (all edges exist).  Uses the networkx.is_simple_path(G, path) function so it
    also ensures the path has no loops!
    :type G: nx.Graph
    """
    ret = False
    try:
        ret = nx.is_simple_path(G, p)
    except KeyError:  # edge not found!
        pass
    return ret


def merge_paths(path1, path2):
    """Merges the two specified paths, which are formatted as returned by get_path()"""

    # Handle path(s) being empty
    if not path1:
        return path2
    if not path2:
        return path1

    # Ensure this is a real path
    if path1[-1] != path2[0]:
        raise ValueError("specified paths don't share a common merging point!  they are %s and %s" % (path1, path2))

    # We just need to remove duplicate node that would appear in the middle of the two paths joined together
    return path1 + path2[1:]


def get_edges_for_path(p):
    """
    Returns the edges in path p using zip
    :param p: a path expressed as an ordered list of nodes
    :type p: list
    :return: ordered list of (src, dst) pairs
    """
    return list(zip(p, p[1:]))


# Simple tests
if __name__ == '__main__':
    npaths = 4

    g = nx.complete_graph(4)
    g.add_edge(0, 4)
    g.add_edge(3, 5)

    ### First test our helper functions
    if not __debug__:
        print "You should run these simple tests without the '-O' flag that optimizes Python" \
              " as we do assert statements to check correctness!"
    assert path_exists(g, [4,0,2,1,3,5])
    assert path_exists(g, [2, 3])
    assert not path_exists(g, [0, 5])
    assert not path_exists(g, [100, 101])

    ### Now test disjoint path algorithms
    print 'test disjoint path algorithms via manual visual inspection...'

    # Need to relabel to strings since we assume nodes are strings
    nx.relabel_nodes(g, {i: str(i) for i in g.nodes()}, copy=False)

    target = '5'
    paths = get_redundant_paths(g, '4', target, npaths)
    print paths

    multi_sources = ['0', '3', '4', '5']
    target2 = '1'
    paths2 = get_multi_source_disjoint_paths(g, multi_sources, target2)
    print paths2

    print 'drawing get_multi_source_disjoint_paths(g, %s, %s) output...' % (multi_sources, target2)
    draw_paths(g, paths2)