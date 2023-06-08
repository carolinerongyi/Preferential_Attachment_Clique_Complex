import numpy as np
from ripser import ripser
from itertools import combinations
from numba import jit


def get_age_matrix(graph):
    """
    Obtain the age matrix of a graph

    OUTPUT
    an n x n matrix, where n is the number of nodes in the graph, representing the age of each edge and node

    INPUT
    graph: an igraph object
    """
    num_nodes = graph.vcount()
    dm = np.zeros((num_nodes, num_nodes))
    edge_list = graph.es
    for edge in edge_list:
        i, j = edge.tuple[0], edge.tuple[1]
        dm[i, j] = max(i, j)

    for i in range(num_nodes):
        dm[i][i] = i
    return dm


@jit(nopython=True)
def translate_PD_to_betti(diagram, max_fil):
    """
    Translate a persistence diagram to a list of betti numbers at each time step

    OUTPUT
    list of betti numbers translated from the persistence diagram

    INPUT
    diagram: persistence diagram of a fixed dimension
    max_fil: the maximum filtration value in the nested sequence; in preferential attachment complex, 
    it is T, the total number of nodes

    """

    betti_increment = np.zeros(max_fil)

    for b, d in diagram:
        betti_increment[int(b)] += 1
        if not np.isinf(d):
            betti_increment[int(d)] -= 1

    betti = np.empty(max_fil)
    betti[0] = betti_increment[0]
    for i in range(1, max_fil):
        betti[i] = betti[i-1] + betti_increment[i]

    return betti


def check_square_appearance(graph, time):
    """
    Check if there's a square in the graph by a given time. If so, return the nodes that form the square

    OUTPUT
    (True, [a, b, c, d]) if there's a square in the graph by a given time, where a, b, c, d are the nodes that form the square
    (False, None) if there's no square in the graph by a given time

    INPUT

    """
    if type(time) == int:
        total_nodes = [i for i in range(time)]
        # try to pick the square with the least sum
        all_combinations = sorted(
            [[a, b, c, d] for a, b, c, d in combinations(total_nodes, 4)], key=sum)
    else:
        all_combinations = [time]
    for combination in all_combinations:
        subgraph = graph.induced_subgraph(combination)
        mat = get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
        if len(dgms[1] == 1):
            return (True, combination)
    return (False, None)


@jit(nopython=True)
def check_node_square_connection(edge_list, node, square, m=7):
    """
    Check if a node is connected to a square in the graph

    OUTPUT
    True if the node is connected to the square, False otherwise

    INPUT
    edge_list: a list of edges in the graph
    node: the node to be checked
    square: the list of nodes that form a square
    m: the number of edges per node
    """
    node_parents = edge_list[(node - 1) * m:node * m, 0]
    for s in square:
        if s not in node_parents:
            return False
    return True


def get_link_matrix(graph, square, node):
    """
    Get the age matrix of the link of a node relative to the square

    OUTPUT
    an n x n matrix, where n is the number of nodes in the graph, representing the age of each edge and node relative to the square

    INPUT
    graph: an igraph object
    square: a list of nodes that form a square
    node: the node to be checked

    """
    link = graph.neighbors(node)
    link = [i for i in link if i <= node]
    num_nodes = len(link)
    dm = np.zeros((num_nodes, num_nodes))
    subg = graph.induced_subgraph(link)

    link = sorted([int(i) for i in link])
    square_indices = [link.index(i) for i in square]

    subg_edge_list = [e.tuple for e in subg.es]
    for edge in subg_edge_list:
        if edge[0] in square_indices and edge[1] in square_indices:
            dm[edge[0], edge[1]] = 1
        else:
            dm[edge[0], edge[1]] = 2

    node_list = [i.index for i in subg.vs]

    for i in node_list:
        dm[i, i] = 2

    for i in square_indices:
        dm[i, i] = 1

    return dm
