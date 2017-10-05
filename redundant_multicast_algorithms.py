__author__ = 'kyle'

import itertools
import networkx as nx
import logging as log

# We aren't using the ILP algorithm currently, so let's not force it as a dependency...
try:
    import pulp
except ImportError:
    pass


class SkeletonList(object):
    """A skeleton list is used to color the edges of a graph either
    blue or red so that two redundant paths (or multicast/spanning
    trees) can always be found from the given source to any
    destination.  If the destination is not 2-reachable, then the
    cut node or link that prevents this will be colored both blue
    and red so these path/tree pairs are maximally redundant.

    It has the following properties:
    TODO

    This data structure and algorithm is based on the
    2013 Bejerano and Koppol (Bell Labs) paper entitled
    "Link-Coloring Based Scheme for Multicast and Unicast Protection"
    """

    # NOTE: subtrees are for holding internal edges.  don't trim the original graph!

    def __init__(self, graph, root, copy_graph=True):
        """The SkeletonList is oriented around a special root node.
        NOTE: if the graph is not directed and you set copy_graph=False,
        graph will be converted to a DiGraph!
        :type graph: nx.Graph
        """

        # ENHANCE: arbitrary orderings to explore:
        # 1) Sets in initial list
        # 2) Order of refinement
        # 3) Pivot node (order of incoming neighbors); also pivot point? Easiest to do right next to anchor set

        # Something about the size of red vs. blue DAGs?

        # TODO: maybe deep copy this to prevent corruption if the graph is changed?  Or just subclass nx.DiGraph...
        if not copy_graph:
            if not graph.is_directed():
                graph.to_directed()
        else:
            graph = nx.DiGraph(graph)
        self.graph = graph
        self.root = root

        self._list = self._get_initial_list(self.graph, self.root)

        assert self.__validate_list()

        self._refine_list()

        assert self.__validate_list()
        # From Observation 2:
        # Anchors are cut nodes for non-anchor nodes in its set.
        # Cutting the anchor would also cut other nodes in this set,
        # so we just check if some other node in this set is a cut node
        # and along the path from anchor to this node.
        # Part a says there are no other incoming links from nodes
        # outside of this set.
        # from networkx.algorithms.connectivity import minimum_st_node_cut
        # for st in self._list[1:-1]:
        #     for n in st.nodes():
        #         if n == self._get_anchor(n):
        #             continue
        #         # Verify cut properities of anchor
        #         cutset = minimum_st_node_cut(self.graph, self.root, n)
        #         assert(len(cutset) == 1)
        #         cutnode = cutset.pop()
        #         anchor = self._get_anchor(n)
        #         assert(cutnode == anchor or cutnode in nx.shortest_path(self.graph, anchor, n))
        #
        #         # Verify lack of other links
        #         preds = self._get_external_predecessors(n)
        #         try:
        #             preds.next()
        #             assert False, "non-anchor nodes should not have external predecessors after initial refinement!"
        #         except StopIteration:
        #             pass

        # Color all the links we can before handling internal links
        self._color_links()

        # We recursively turn any non-refinable subtree sets into their own
        # skeleton lists in order to color their 'internal' links correctly
        self._recurse()

        # Turning this off for now as it's really slow for bigger topologies
        # assert self.__validate_coloring()

    def _get_initial_list(self, graph, root):
        """We initialize a skeleton list by placing the root alone in the
        first and last sets.  Each of its successors becomes the anchor
        and root for a 'subtree set' of an initially-computed spanning tree.
        These trees make up the other initial 'set' entries in the list.
        """

        root_subtree = nx.DiGraph()
        root_subtree.add_node(root)
        self._update_bookkeeping(root_subtree, root)
        _list = [root_subtree]

        # We don't care about weights for this tree so just use BFS
        # TODO: optionally care about weights?  this would require subgraphing
        # anywhere we create a new graph using e.g. bfs_tree
        tree = nx.bfs_tree(graph, root)

        # Need a list as generator will break during iteration due to modifications
        # ORDER is arbitrary here, as is the tree construction method
        for n in list(tree.successors(root)):
            subtree = self._trim_subtree(tree, n)
            _list.append(subtree)

        return _list + [root_subtree]

    def _refine_list(self):
        """We need to iteratively refine the skeleton list until we can
        no longer do so.  At this point, it's either a complete
        ordering of the nodes, or some of the anchors are cut nodes
        of their predecessors within their set."""

        # We'll refine each subtree completely before moving to the next.
        # Otherwise, we might have a node that doesn't appear refinable
        # currently but will once its predecessor moves out of its subtree.
        # NOTE: the order we do them in is arbitrary
        subtrees_to_refine = self._list[1:-1]

        while len(subtrees_to_refine) > 0:
            next_subtree = subtrees_to_refine.pop()

            # Go through the subtree and trim off as many refinable nodes
            # as possible, taking care to skip over the root and any
            # already-visited nodes.
            # A node is refinable if it's a non-anchor node in this set
            # having an incoming neighbor in a different set;
            # a refinable node is 2-reachable.
            # NOTE: order we look at the nodes is arbitrary, but we use
            # DFS to try and split up this subtree into as many other
            # subtrees as possible during this iteration.
            anchor = self._get_anchor(next_subtree)
            # If this set had no refinable nodes, we're done with it;
            # otherwise we may have to check it again since the list changed.
            this_set_unrefinable = True
            for next_node in list(i for i in nx.dfs_postorder_nodes(next_subtree, anchor))[:-1]:
                assert(anchor != next_node)
                # already trimmed off this one
                if self._get_anchor(next_node) != anchor:
                    continue

                # Find an incoming neighbor not in this set if possible
                # NOTE: ORDER is arbitrary here
                external_predecessors = self._get_external_predecessors(next_node)
                try:
                    pred = external_predecessors.next()
                except StopIteration:
                    pred = None
                if pred is not None:
                    # If we found one, trim off this node and its subtree
                    # then add it as a new subtree set at the proper
                    # location within the skeleton list w.r.t. the
                    # predecessor we found.
                    new_subtree = self._trim_subtree(next_subtree, next_node)
                    assert(all(self._get_anchor(n) == next_node for n in new_subtree.nodes()))
                    assert(self._get_anchor(next_node) == next_node)
                    this_set_unrefinable = False

                    # We need to insert the new subtree set between this one
                    # and its predecessors'.  Easiest is to insert it to the
                    # right or left of the current subtree set, though the
                    # actual location is arbitrary as long as it satisfies ordering.
                    pred_index = self._get_index(pred)
                    # NOTE: depending on implementation details, this could change
                    # in between iterations of this loop.
                    this_index = self._get_index(anchor)

                    if pred_index < this_index:
                        self._insert_subtree(new_subtree, this_index)
                    else:
                        self._insert_subtree(new_subtree, this_index+1)

                    # TODO: this may be unnecessary as we do post-order, meaning anything refinable will be
                    if new_subtree.number_of_nodes() > 1:
                        subtrees_to_refine.append(new_subtree)

                # else it's not a refinable node; not 2-reachable

            if not this_set_unrefinable and next_subtree.number_of_nodes() > 1:
                subtrees_to_refine.append(next_subtree)

    def _update_bookkeeping(self, tree, root):
        for n2 in tree.nodes():
            self._set_anchor(n2, root)
            self._set_subtree_for_node(n2, tree)

    def _get_node_data(self, node):
        return self.graph.node[node]

    def _get_edge_data(self, edge):
        return self.graph[edge[0]][edge[1]]

    def _set_edge_color(self, edge, color):
        self._get_edge_data(edge)['skeleton_list_color'] = color

    def _get_edge_color(self, edge):
        return self._get_edge_data(edge)['skeleton_list_color']

    def _get_subtree_for_node(self, node):
        return self._get_node_data(node)['skeleton_list_subtree']

    def _set_subtree_for_node(self, node, subtree):
        self._get_node_data(node)['skeleton_list_subtree'] = subtree

    def _get_anchor(self, node_or_tree):
        # the anchor of all nodes in the subtree is the same,
        # which is considered the anchor of the subtree itself
        if isinstance(node_or_tree, nx.Graph):
            node_or_tree = node_or_tree.nodes().next()
        # TODO: this may not work with non-anchor nodes
        if isinstance(node_or_tree, SkeletonList):
            node_or_tree = node_or_tree.root
        return self._get_node_data(node_or_tree)['skeleton_list_anchor']

    def _is_anchor(self, node):
        return self._get_anchor(node) == node

    def _set_anchor(self, node, anchor):
        self._get_node_data(node)['skeleton_list_anchor'] = anchor

    def _get_index(self, node_or_tree):
        if not isinstance(node_or_tree, nx.Graph):
            return self._get_index(self._get_subtree_for_node(node_or_tree))
        # TODO: could do some trickery with numbering and use a linked list:
        # instead of changing all the indexes like we would with an array list,
        # we just average the 'indices' of the adjacent linked nodes we add this
        # between.  Makes this op O(1) (though needs mods) instead of O(V).
        return self._list.index(node_or_tree)

    def _get_external_predecessors(self, node):
        """Returns predecessors (incoming edges) that are not in the same subtree set."""
        return (p for p in self.graph.predecessors(node)\
                if self._get_anchor(p) != self._get_anchor(node))

    def _trim_subtree(self, tree, root):
        """Trims off and returns a subtree rooted at root. Updates all
        necessary internal state.
        :param tree:
        :type tree: nx.DiGraph
        :param root:
        :return: subtree rooted at root
        """
        subtree = nx.dfs_tree(tree, root)
        self._update_bookkeeping(subtree, root)
        tree.remove_nodes_from(subtree.nodes())
        return subtree

    def _insert_subtree(self, tree, index):
        self._list.insert(index, tree)

    def _color_links(self):
        """Color all forward edges red and backward edges blue. A forward
        edge points to a higher-indexed node."""

        # See below special case
        nblue_root_links = 0
        nred_root_links = 0

        # We only need to color edges to/from anchors as internal links
        # will be handled by the recursion
        for dst_idx in range(1, len(self._list) - 1):
            anchor = self._get_anchor(self._list[dst_idx])
            # SPECIAL CASE: For each r,v link from the root to some other node v:
            # If no other incoming neighbors, it's a cut link and gets both red and blue;
            # else if all incoming neighbors are after (before) color it red (blue);
            # else choose some arbitrary criteria (we balance them evenly).
            root_link = None
            pred_before = False
            pred_after = False

            for pred in self._get_external_predecessors(anchor):
                # Need to handle root later
                if pred == self.root:
                    root_link = (pred, anchor)
                    continue

                src_idx = self._get_index(pred)
                color = None
                if src_idx < dst_idx:
                    pred_before = True
                    color = 'red'
                elif src_idx > dst_idx:
                    pred_after = True
                    color = 'blue'
                self._set_edge_color((pred, anchor), color)

            if root_link is not None:
                color = None
                # No other incoming neighbors: cut link
                if not pred_after and not pred_before:
                    color = 'red-blue'
                # Arbitrary criteria: balance them
                elif pred_before and pred_after:
                    if nred_root_links <= nblue_root_links:
                        color = 'red'
                        nred_root_links += 1
                    else:
                        color = 'blue'
                        nblue_root_links += 1
                # All after
                elif pred_after:
                    color = 'red'
                # All before
                elif pred_before:
                    color = 'blue'
                self._set_edge_color(root_link, color)

        # Specially handle v,r links, which the paper doesn't consider
        # We adopt the arbitrary choice to just disable these links since they won't be
        # used in our multicast scenario anyway.
        for pred in self.graph.predecessors(self.root):
            self._set_edge_color((pred, self.root), 'black')
            # TODO: figure out how to properly assign a real color?

    def _recurse(self):
        """We need to recursively apply the skeleton list procedure on all
        subtree sets of size > 1 because their anchor nodes are cut nodes
        for the non-anchors."""

        # We won't resize the list at this point, so we can work with indexes directly
        for idx in range(1, len(self._list) - 1):
            this_set = self._list[idx]
            if this_set.number_of_nodes() == 1:
                continue

            # Recursively turn this subtree into a skeleton list to color it,
            # but need to pass the recursive call a sugraph, which it should
            # not copy, or else we won't get the color on our edges here.
            # TODO WARNING: this may corrupt our data structures by changing
            # the anchor of currently non-anchor nodes.  This shouldn't matter
            # unless we try to handle dynamic modifications to the graph.
            # To fix this, need to store anchors and subtree pointers in the
            # SkeletonList object itself, but share coloring globally.
            anchor = self._get_anchor(this_set)
            subg = self.graph.subgraph(this_set.nodes())
            skelist = SkeletonList(subg, anchor, copy_graph=False)
            self._list[idx] = skelist
            # TODO: if we want to keep around a SkeletonList to use for dynamic
            # topologies, we should be able to handle when a subtree set has
            # become a SkeletonList itself: maybe we subclass nx.DiGraph?
            assert this_set.number_of_edges() == 0 or \
                   all(d.get('skeleton_list_color', False) for u,v,d in skelist.graph.edges(data=True)),\
                "all internal links should be colored at this point"
        assert all(d.get('skeleton_list_color', False) for u,v,d in self.graph.edges(data=True)), "ALL links should be colored at this point"
        # assert self.__validate_list()  # dies because of above corruption warning

    def get_red_graph(self):
        return self.graph.edge_subgraph((u,v) for u,v,d in self.graph.edges(data=True)\
            if 'red' in d['skeleton_list_color'])

    def get_blue_graph(self):
        return self.graph.edge_subgraph((u,v) for u,v,d in self.graph.edges(data=True)\
            if 'blue' in d['skeleton_list_color'])

    # TODO: handle dynamic topologies by subclassing nx.DiGraph and overriding some methods

    # Below functions should only be used for testing purposes
    def __validate_list(self):
        non_source_sets = self._list[1:-1]
        # every non-source anchor has an incoming neighbor
        # both before and after it in the skeleton list.
        for anchor in (self._get_anchor(t) for t in non_source_sets):
            found_after = False
            found_before = False
            my_idx = self._get_index(anchor)
            for pred in self._get_external_predecessors(anchor):
                pred_idx = self._get_index(pred)
                if pred_idx < my_idx:
                    found_before = True
                # HACK: root is in 2 places!
                if pred == self.root:
                    pred_idx = len(self._list) - 1
                if pred_idx > my_idx:
                    found_after = True

            assert(found_after and found_before)

        # every non-source set is pair-wise disjoint with every other set
        for s1, s2 in itertools.combinations(non_source_sets, 2):
            assert(len(set(s1.nodes()).intersection(s2.nodes())) == 0)

        # directed path from anchor to every other node in its set
        assert(all(nx.is_directed_acyclic_graph(t) for t in non_source_sets))

        # first and last sets are always just the root
        assert(self.root in self._list[0] and self.root in self._list[-1] and\
               self._list[0].number_of_nodes() == 1 and self._list[-1].number_of_nodes() == 1)

        return True

    def __validate_coloring(self):
        log.debug("validating red-blue coloring; this could take a while (forever) if your graph is too big")
        # All red paths to a vertex should be disjoint from all blue paths
        # to the same vertex, except for red-blue links and their incident nodes
        red_dag = self.get_red_graph()
        blue_dag = self.get_blue_graph()
        source = self.root

        for dst in self.graph.nodes():
            if dst == source:
                continue

            red_paths = list(nx.all_simple_paths(red_dag, source, dst))
            red_nodes = set(n for p in red_paths for n in p)
            red_edges = set(e for p in red_paths for e in zip(p, p[1:]))
            blue_paths = list(nx.all_simple_paths(blue_dag, source, dst))
            blue_nodes = set(n for p in blue_paths for n in p)
            blue_edges = set(e for p in blue_paths for e in zip(p, p[1:]))

            redblue_edges = red_edges.intersection(blue_edges)
            redblue_nodes = red_nodes.intersection(blue_nodes)
            redblue_nodes.remove(source)
            redblue_nodes.remove(dst)

            assert all(self._get_edge_color(e) == 'red-blue' for e in redblue_edges),\
                "invalid coloring: non cut link shared by red and blue paths!"
            # every shared node has at least one incident cut link
            # TODO: finish this?  unclear it's necessary as it just validates consistency of coloring not actual correctness of properties
            # assert all(any(self._get_edge_color(e) == 'red-blue' for e in
            #                list(self.graph.successors(n)) + list(self.graph.predecessors(n)))
            #            for n in redblue_nodes), "invalid coloring of nodes: shares a non-cut node!"

            # verify each red-blue edge or node is a cut edge/node
            for cut_node in redblue_nodes:
                g = self.graph.subgraph(n for n in self.graph.nodes() if n != cut_node)
                # could induce an empty (or near-empty) graph
                if source not in g or dst not in g:
                    continue
                assert not nx.has_path(g, source, dst), "invalid coloring: non cut node shared by red and blue paths!"
            for cut_link in redblue_edges:
                g = self.graph.edge_subgraph(e for e in self.graph.edges() if e != cut_link)
                # could induce an empty (or near-empty) graph
                if source not in g or dst not in g:
                    continue
                assert not nx.has_path(g, source, dst), "invalid coloring: non cut link shared by red and blue paths!"
        # draw_overlaid_graphs(self.graph, [red_dag, blue_dag])

        return True

    def print_list(self):
        print [tuple(t.nodes() if isinstance(t, nx.Graph) else t.graph.nodes()) for t in self._list]




def ilp_redundant_multicast(topology, source, destinations, k=2, get_lower_bound=False):
    # ENHANCE: use_multicommodity_flow=False,
    """Uses pulp and our ILP formulation to create k redundant
    multicast trees from the source to all the destinations on the given topology.
    NOTE: this assumes nodes in the topology are represented as strings!
    :param get_lower_bound: if True, simply returns the lower bound of the
     overlap for k trees as computed by the relaxation
    """

    relaxed = get_lower_bound
    var_type = pulp.LpContinuous if relaxed else pulp.LpBinary

    # Extract strings to work with pulp more easily
    # edges = [edge_to_str(e) for e in topology.edges()]
    edges = list(topology.edges())
    vertices = [str(v) for v in topology.nodes()]
    multicast_trees = ["T%d" % i for i in range(k)]

    # First, convert topology and parameters into variables
    # To construct the multicast trees, we need TE variables that determine
    # if edge e is used for tree t
    edge_selection = pulp.LpVariable.dicts("Edge", (edges, multicast_trees), 0, 1, var_type)

    # OBJECTIVE FUNTION: sums # shared edges pair-wise betweend all redundant trees
    # That is, the total cost is sum_{links}{shared^2} where shared = # trees sharing this link
    # To linearize the objective function, we need to generate ET^2 variables
    # that count the overlap of the trees for a given edge
    # NOTE: since the pair-wise selection of the trees is symmetrical,
    # we only look at the lower diagonal half of the 'matrix' to reduce # variables
    pairwise_edge_tree_choices = [(e, t1, t2) for e in edges for t1 in multicast_trees for t2 in multicast_trees if t2 <= t1]
    # print "\n".join([str(a) for a in pairwise_edge_tree_choices])
    overlap_costs = dict()
    for e, t1, t2 in pairwise_edge_tree_choices:
        overlap_costs.setdefault(e, dict()).setdefault(t1, dict())[t2] = \
            pulp.LpVariable("Overlap_%s,%s_%s_%s" % (e[0], e[1], t1, t2), 0, 1, var_type)

    # To ensure each destination is reached by all the multicast trees,
    # we need a variable to determine if an edge is being used on the path
    # to a particular destination for a particular tree.
    # NOTE: this edge-based variable is DIRECTED despite the path being undirected
    #   it means that a unit of flow is selected from u to v
    _flipped_edges = zip(*reversed(zip(*edges)))
    _flipped_edges.extend(edges)
    _sources, _dests = zip(*_flipped_edges)
    path_selection = pulp.LpVariable.dicts("Path", (_sources, _dests, destinations, multicast_trees), 0, 1, var_type)

    # TODO: set the starting values of some variables (destination flows and edges?)
    # using var.setInitialValue

    problem = pulp.LpProblem("Redundant Multicast Topology", pulp.LpMinimize)

    # OBJECTIVE FUNTION:
    # Since we're only computing the lower-half diagonal of this, we need to double the proper lower half
    problem += pulp.lpSum([overlap_costs[e][t][t] for e in edges for t in multicast_trees]) + \
        2 * pulp.lpSum([overlap_costs[e][t1][t2] for e, t1, t2 in pairwise_edge_tree_choices if t1 != t2]),\
               "Pair-wise overlap among trees"

    # CONSTRAINTS:

    # These ~1.5*E*T^2 constraints help linearize the overlap_costs components of the objective function.
    # It counts the lower-diagonal overlap, multiplying the resulting sum by 2, and adding the sum over diagonal
    for e, t1, t2 in pairwise_edge_tree_choices:
        # This ensures the overlap is 0 if the edge not selected by a tree
        problem += overlap_costs[e][t1][t2] <= edge_selection[e][t1],\
                   "OverlapRequirement1_%s_%s_%s" % (e, t1, t2)
        # LOWER DIAGONAL MATRIX FORMULATION: same as above constraint for t2
        if t1 != t2:
            problem += overlap_costs[e][t1][t2] <= edge_selection[e][t2],\
                       "OverlapRequirement2_%s_%s_%s" % (e, t1, t2)
        # FULL MATRIX FORMULATION:
        # Since we have an undirected graph and are counting the overlap 'twice',
        # simply ensure the overlap matrix is mirrored
        # problem += overlap_costs[e][t1][t2] == overlap_costs[e][t2][t1], \

        # This ensures the cost is 1 if overlap
        problem += overlap_costs[e][t1][t2] >= edge_selection[e][t1] + edge_selection[e][t2] - 1,\
                   "OverlapRequirement3_%s_%s_%s" % (e, t1, t2)

    # These 3 flow constraints ensure each destination has a path to it on each tree,
    # i.e. each multicast tree covers the terminals.
    # NOTE: we have an undirected graph but the flow constraints are directed
    for t in multicast_trees:
        for d in destinations:
            # First, the source and destination should have 1 total unit
            # of outgoing and incoming flow, respectively
            problem += pulp.lpSum([path_selection[source][v][d][t] for v in topology.neighbors(source)]) == 1,\
                       "SourceFlowRequirement_%s_%s" % (t, d)
            problem += pulp.lpSum([path_selection[v][d][d][t] for v in topology.neighbors(d)]) == 1,\
                       "DestinationFlowRequirement_%s_%s" % (t, d)
            # Then, every other 'interior' vertex should have a balanced in and out flow.
            # Really, both should be either 0 or 1 each
            for v in vertices:
                if v == d or v == source:
                    continue
                problem += pulp.lpSum([path_selection[u][v][d][t] for u in topology.neighbors(v) if u != v]) ==\
                           pulp.lpSum([path_selection[v][u][d][t] for u in topology.neighbors(v) if u != v]),\
                           "InteriorFlowRequirement_%s_%s_%s" % (t, d, v)

    # This constraint simply ensures an edge counts as selected on a tree
    # when it has been selected for a path to a destination along that tree
    # in order to satisfy the flow constraints.
    # NOTE: if the edge is path-selected in either direction, it counts as selected for the tree
    # NOTE: this constraint implies that we cannot have path_selection[u][v][d][t] = 1 = path_selection[v][u][d][t]
    for e in edges:
        for t in multicast_trees:
            for d in destinations:
                u,v = e
                # problem += path_selection[e][d][t] <= edge_selection[e][t],\
                problem += path_selection[u][v][d][t] + path_selection[v][u][d][t] <= edge_selection[e][t],\
                           "EdgeSelectionRequirement_%s_%s_%s" % (e,d,t)

    # Now we can solve the problem
    problem.solve()

    log.info("#Variables: %d" % len(problem.variables()))
    log.info("#Constraints: %d" % len(problem.constraints))
    log.info("Status: %s" % pulp.LpStatus[problem.status])
    cost = pulp.value(problem.objective)
    log.info("Cost: %f" % cost) # should be 7 for C(4) + 2 graph with 1 dest

    if get_lower_bound:
        return cost
    else:
        # Lastly, construct multicast trees from the edge_selection variables and return the results
        # We use subgraph in order to ensure the attributes remain
        final_trees = []
        for t in multicast_trees:
            res = topology.edge_subgraph(e for e in edges if edge_selection[e][t].value())
            final_trees.append(res)
            assert nx.is_tree(res) and all(nx.has_path(res, source, d) for d in destinations)

        return final_trees


# tests
if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    # TEST SKELETON LIST
    # This test graph comes from Fig. 2 of the paper
    g = nx.DiGraph([('r','x'), ('r','w'), ('r','a'), ('a','b'), ('a','c'),
                    ('b','x'), ('c','w'), ('b','c'), ('c','b'),
                    ('x','y'), ('w','z'), ('b','d'), ('c','d'), ('d','e'),
                    ('e','y'), ('e','z')])
    root = 'r'

    # enhance that graph by adding some more edges so graph is deeper
    # and we can see > 2 refinements being made before recursion
    g.add_edge('x','m')
    g.add_edge('y','m')

    # g = nx.complete_graph(8)  # about 40 gets slowwwww
    # root = 0

    # filename = 'campus_topo_20b-8h-3ibl.json'
    # filename = 'campus_topo_200b-20h-20ibl.json'  # this takes a while!
    # from networkx.readwrite import json_graph
    # import json
    # with open(filename) as f:
    #     data = json.load(f)
    #     g = json_graph.node_link_graph(data)
    # root = 's0'

    sl = SkeletonList(g, root)

    print "final list:",
    sl.print_list()

    red = sl.get_red_graph()
    blue = sl.get_blue_graph()
    # intersection should only be cut-links
    green = nx.algorithms.intersection(red, blue)

    assert nx.is_directed_acyclic_graph(red) and nx.is_directed_acyclic_graph(blue)

    from dsm_networkx_algorithms import draw_overlaid_graphs
    draw_overlaid_graphs(g, [red, blue, green])

    # recursively partition them: what happens?
    redsl = SkeletonList(red, root)
    bluesl = SkeletonList(blue, root)
    redred = redsl.get_red_graph()
    redblue = redsl.get_blue_graph()
    bluered = bluesl.get_red_graph()
    blueblue = bluesl.get_blue_graph()

    draw_overlaid_graphs(g, [redred, redblue, bluered, blueblue])
