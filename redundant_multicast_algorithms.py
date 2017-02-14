__author__ = 'kyle'

import networkx as nx
import pulp


def ilp_redundant_multicast(topology, source, destinations, k=2):
    """Uses pulp and our ILP formulation to create k redundant
    multicast trees from the source to all the destinations on the given topology.
    NOTE: this assumes nodes in the topology are represented as strings!
    """

    # Extract strings to work with pulp more easily
    # edges = [edge_to_str(e) for e in topology.edges()]
    edges = list(topology.edges())
    vertices = [str(v) for v in topology.nodes()]
    multicast_trees = ["T%d" % i for i in range(k)]

    # First, convert topology and parameters into variables
    # To construct the multicast trees, we need TE variables that determine
    # if edge e is used for tree t
    edge_selection = pulp.LpVariable.dicts("Edge", (edges, multicast_trees), 0, 1, pulp.LpInteger)

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
            pulp.LpVariable("Overlap_%s,%s_%s_%s" % (e[0], e[1], t1, t2), 0, 1, pulp.LpInteger)

    # To ensure each destination is reached by all the multicast trees,
    # we need a variable to determine if an edge is being used on the path
    # to a particular destination for a particular tree.
    # NOTE: this edge-based variable is DIRECTED despite the path being undirected
    #   it means that a unit of flow is selected from u to v
    _flipped_edges = zip(*reversed(zip(*edges)))
    _flipped_edges.extend(edges)
    _sources, _dests = zip(*_flipped_edges)
    path_selection = pulp.LpVariable.dicts("Path", (_sources, _dests, destinations, multicast_trees), 0, 1, pulp.LpInteger)

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
    # the overlap, multiplying the resulting sum by 2, and adding the sum over diagonal?
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

    print "#Variables:", len(problem.variables())
    print "#Constraints:", len(problem.constraints)
    print "Status:", pulp.LpStatus[problem.status]
    print "Cost:", pulp.value(problem.objective) # should be 7 for C(4) + 2 graph with 1 dest

    # Lastly, construct multicast trees from the edge_selection variables and return the results
    # TODO: get the 'data' back into the nodes?
    final_trees = []
    for t in multicast_trees:
        res = nx.Graph()
        final_trees.append(res)
        for e in edges:
            if edge_selection[e][t].value():
                res.add_edge(*e)

    return final_trees

