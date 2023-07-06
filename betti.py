import numpy as np
from ripser import ripser
from itertools import combinations
from numba import jit


def get_age_matrix(graph):
    """
    Obtain the age matrix of a graph whose vertices are indexed by integers 0, ..., n-1.
    The graph is conceived to be built by adding vertices one by one, and the
    vertices are indexed by the time it is added to the graph.
    An edge is added to the graph as soon as both endpoints are added to the 
    graph.

    The "age" of a vertex is its index.
    The "age" of an edge is the larger of its two endpoints.

    OUTPUT
    an n x n upper triagnular matrix, where n is the number of nodes in the graph
    The diagonal entries are the "ages" of the vertices.
    The offdiagonal entries are the "ages" of the edges.

    INPUT
    graph: an igraph object whose vertices are indexed by integers 0, ..., n-1
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
    Translate a persistence diagram to a list of betti numbers at each time step (or filtration value)
    At time t, the Betti number is the number of independent homology classes
    that have been born but not been dead. In other words, it is the number of
    points in the upper-left quadrant whose lower-right corner is (t, t).

    OUTPUT
    a numpy array of betti numbers at each time instant

    INPUT
    diagram: a persistence diagram of a fixed dimension
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


def betti_numbers_of_links(graph, num_nodes, m, maxdim):
    """
    Return the Betti numbers of the precedent links of the node, where the 
    precedent link of a vertex if the subcomplex formed by all nodes connected
    the vertex with a smaller index.

    It is useful for getting upper and lower bounds of the Betti numbers

    OUTPUT
    a (maxdim + 1) x T array, entry (q, i) is Betti q of the precedent link of 
    vertex i

    INPUT
    graph: igraph object
    num_nodes: number of nodes in the graph
    m: number of edges per new nodes
    maxdim: the maximum dimension at which the Betti numbers are computed
    """

    edge_list = np.array([e.tuple for e in graph.es])

    start = 0
    end = m
    betti_nums = np.zeros((maxdim + 1, num_nodes))
    current_node = 1

    for current_node in range(1, num_nodes):
        start = (current_node - 1) * m
        end = current_node * m
    # while start < len(edge_list) and end <= len(edge_list):

        # find the link of each node
        link_list = edge_list[start:end, 0]
        link_list = np.unique(link_list)

        # generate subgraph for the link
        subgraph = graph.induced_subgraph(link_list)

        # count the betti numbers
        mat = get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=maxdim)['dgms']

        for dim in range(maxdim):
            for k, j in dgms[dim]:
                if j == np.inf:
                    betti_nums[dim, current_node] += 1

        # current_node += 1
        # start += m
        # end += m

    return betti_nums


def check_square_appearance(graph, time):
    """
    Check if there's a hollow square in the graph by a given time,
    of if a specific tuple of 4 nodes form a hollow square in the graph.
    If so, return as well the nodes that form one such square.
    A hollow square is a square without any diagonals.

    OUTPUT
    (True, np.array([a, b, c, d])) if a, b, c, d forms a desired square
    (False, None) if there is no desired square

    INPUT
    graph: an igraph object whose vertices are indexed by 0, ..., n-1
    time: Either a specific time limit before which the square is to be found
    Or the specific square

    For example, if check_square_appearance(graph, 20) returns (True, [0, 4, 5, 7]),
    this means there is a square in the graph formed by the first 20 nodes, 
    and [0, 4, 5, 7] is one such square.
    If check_square_appearance(graph, 20) returns (False, None),
    the first 20 nodes do not form any squares.

    If check_square_appearance(graph, [1, 4, 6, 20]) returns (True, [1, 4, 6, 20]),
    then [1, 4, 6, 20] is a square. If it returns (False, None), then it does 
    not form a square in the graph.

    """
    if type(time) == int:
        for last_node_in_square_to_test in range(3, time + 1):
            for a, b, c in combinations(range(last_node_in_square_to_test), 3):
                combination = [a, b, c, last_node_in_square_to_test]
                boo = check_one_combination(graph, combination)
                if boo:
                    return boo, np.array(combination)

        # total_nodes = [i for i in range(time)]
        # # try to pick the square with the least sum
        # all_combinations = sorted(
        #     [[a, b, c, d] for a, b, c, d in combinations(total_nodes, 4)], key=sum)
    else:
        combination = [time]
        boo = check_one_combination(graph, combination)
        if boo:
            return boo, np.array(combination)
    return (False, None)
    # for combination in all_combinations:
    #     subgraph = graph.induced_subgraph(combination)
    #     mat = get_age_matrix(subgraph)
    #     dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
    #     if len(dgms[1] == 1): # This is an innocuous bug.
    #                           # It should be len(dgms[1]) == 1
    #                           # But since positive integers are regarded as True
    #                           # and since the only possible way to have a hole
    #                           # with four points is to have a square
    #                           # The two codes behave exactly the same
    #         return (True, np.array(combination))
    # return (False, None)


def check_one_combination(graph, combination):
    """
    helper function for check_square_appearance(graph, time)

    test whether the tuple of nodes in combination forms a hollow square

    """
    subgraph = graph.induced_subgraph(combination)
    mat = get_age_matrix(subgraph)
    dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
    return len(dgms[1]) == 1

@jit(nopython=True)
def check_node_square_connection(edge_list, m, node, square):
    """
    Check if a node is connected to a square in the graph

    OUTPUT
    True if the node is connected to the square, False otherwise

    INPUT
    edge_list: a list of edges in the graph
               This list must be formed by 
               edge_list = np.array([e.tuple for e in graph.es]),
               where graph must be generated simulator_pa.pa_generator
               because we exploit a specific structure of the graph

    m: the number of edges per new node    
    node: the node to be checked
    square: the list of nodes that form a hollow square
    """
    
    node_parents = edge_list[(node - 1) * m:node * m, 0]
    for s in square:
        if s not in node_parents:
            return False
    return True


def get_link_matrix(graph, square, node):
    """
    helper function of second_summand_lower_bound

    Get the age matrix of the link of a node relative to the square
    so that we can compute the nullity in the definition of $\hat{b_{IK}}$ in
    Section 8. See the paper and the Jupyter notebook for details.

    OUTPUT
    an n x n matrix, where n is the number of nodes in the link, representing the age of each edge and node relative to the square
    The "relaive age" of nodes and edges in the square are 1.
    The "relative age" of nodes and edges in the square are 2.

    INPUT
    graph: an igraph object
    square: a list of nodes that form a hollow square
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


@jit(nopython=True)
def first_summand_lower_bound(edge_list, num_nodes, m, square):
    """
    Compute the $\hat \ell^{(t)}$ in Section 8.
    See the paper and the Jupyter notebook for details.

    OUTPUT
    An array of length num_nodes, entry t is the value of $\hat \ell^{(t)}$ 

    INPUT
    edge_list: output of np.array([e.tuple for e in graph.es]), where
               graph is an output of pa_generator
    num_nodes: number of nodes in the graph
    m: number of new edges per new node
    square: a list of nodes that form a square

    """

    first_summand = np.zeros(num_nodes)  # numba does not like integer arrays
    found_one_flag = 0
    latest_node_in_square = max(square)
    for i in range(latest_node_in_square + 1, num_nodes):
        fill = check_node_square_connection(edge_list, m, i, square)
        if fill:
            if found_one_flag == 0:  # The first filling does not count
                found_one_flag = 1
            else:
                first_summand[i] = 1

    return first_summand


def second_summand_lower_bound(graph, num_nodes, square, first_summand):
    """
    Compute the nullity in the definition of $\hat{b_{IK}}$ in
    Section 8. See the paper and the Jupyter notebook for details.
    The nullity is positive when there is a point in the persistence diagram
    with death time 2

    OUTPUT
    An array of length num_nodes, entry t is the value of $\hat{b_{IK}}^{(t)}$ 

    INPUT
    graph: an igraph object
    num_nodes: number of nodes in the graph
    square: a list of nodes that form a hollow square
    first_summand: output of first_summand_lower_bound

    """
    latest_node_in_square = max(square)
    second_summand = np.zeros(num_nodes, dtype=int)
    for i in range(latest_node_in_square + 1, num_nodes):
        if first_summand[i] >= 1:
            mat = get_link_matrix(graph, square, i)
            dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
            second_summand[i] = int(
                any([pt[1] == 2 for pt in dgms[1]])
            )
    return second_summand


def betti2_lower_bound(graph, time=20, first_summand=None, second_summand=None, links_betti_nums=None):
    """
    Return the lower bound of Betti 2.
    Summands of individual terms are output as well.

    OUTPUT
    output 1: an array of length num_nodes, entry t is a lower bound of
              Betti 2 of the subcomplex consisting of the first t nodes
    output 2: output of first_summand_lower_bound, see comments therein
    output 3: output of second_summand_lower_bound, see comments therein
    output 4: output of betti_numbers_of_links, see comments therein

    INPUT
    graph: igraph object
    time: the algorithm tries to find a square whose node indices are at most
          time (if time is an integer) or are precisely entries in time (if time
          is a 4-tuple of increasing nonnegative integers).
          If such a square does not exist, the first two terms are zero.
    first_summand, second_summand, links_betti_nums: optional inputs of 
        first_summand_lower_bound, second_summand_lower_bound and 
        betti_numbers_of_links. They may be supplied if these quantities have
        been saved so as to avoid repeated computation
    """

    edge_list = np.array([e.tuple for e in graph.es])
    num_nodes = graph.vcount()
    m = int(len(edge_list)/(num_nodes - 1))

    # find if there's a square in the first 20 nodes
    boo, square = check_square_appearance(graph, time)

    if not boo:
        first_term = np.zeros(num_nodes, dtype=int)
    else:
        # latest_node_in_square = max(square)
        if first_summand is None:
            first_summand = first_summand_lower_bound(
                # edge_list, num_nodes, m, square, latest_node_in_square)
                edge_list, num_nodes, m, square)
            first_summand = first_summand.astype(int)
        first_term = np.cumsum(first_summand)

    if not boo:
        second_term = np.zeros(num_nodes, dtype=int)
    else:
        if second_summand is None:
            second_summand = second_summand_lower_bound(
                graph, num_nodes, square, first_summand)
        second_term = np.cumsum(second_summand)

    if links_betti_nums is None:
        links_betti_nums = betti_numbers_of_links(
            graph, num_nodes, m, maxdim=2)

    third_term = np.cumsum(links_betti_nums[2, :])

    return first_term - second_term - third_term, first_summand, second_summand, links_betti_nums
