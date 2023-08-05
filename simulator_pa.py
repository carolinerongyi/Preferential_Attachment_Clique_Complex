#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:55:48 2022

@author: CarolineHe
"""
import numpy as np
import igraph as ig
from numba import jit


def pa_generator(num_nodes, num_edges_per_new_node, delta, seed, polya_flag=False, sorted_flag=False, early_exit=None, python_object_flag=True):
    """
    build a graph with the prefential attachment model in 
    [Garavaglia 2019] (www.doi.org/10.1017/apr.2019.36)
    Each node are connected to (num_edges_per_new_node) earlier nodes.
    The probability of the connection is proportional to the degree of the 
    earlier node plus delta.
    When delta is large, the rich-gets-richer effect is not as strong.
    See the paper for details.

    OUTPUT
    a multigraph (with multiple edges) constructed with the model
    either in the form of an igraph graph object (if python_object_flag == 1) 
    or as an edge list (if python_object_flag == 0)

    INPUT
    num_nodes
        number of nodes in the graph
    num_edges_per_new_node
        number of edges attached to the new node each time
    delta
        parameter to control the rich-gets-richer effect
        For smaller (more negative) delta, the effect is stronger
        The power of of the tail of the degree distribution should be
        tau = 3 + delta / m
        We choose -m < delta < 0 to ensure 2 < tau < 3.

    seed 
        seed for the random number geneator
    polya_flag
        binary, whether to use Definition 2 in [Garavaglia 2019] for graph
        generation to speed up the codes
    sorted_flag
        binary, relevant only if polya_flag == 1
        whether edges with the same child are to be sorted by parents in the 
        output graph, this is useful for counting the edges' multiplicity
        If polya_flag == 0, edges will NOT be sorted
    early_exit
        nonnegative integer, relevant only if polya_flag == 1
        terminate the graph construction once the number of nodes reaches
        early_exit
        While the distribution is the same as that of the graph with the same 
        number of nodes, this parameter ensures that the same graph is obtained
        when the same seed is applied. See example below.
    python_object_flag
        binary, whether to output the graph as an igraph graph object (as 
        opposed to an edge list)

    EXAMPLE

    graph1 = pa_generator(10, 5, -3, seed = 0, polya_flag=True)
    graph2 = pa_generator(20, 5, -3, seed = 0, polya_flag=True)
    graph3 = pa_generator(20, 5, -3, seed = 0, polya_flag=True, early_exit = 10)
    graph3 is the indced subgraph of graph2 on the first 10 nodes.
    It is DIFFERENT from graph1 because the graph is sampled differently when 
    the number of nodes.

    """
    assert (num_nodes > 1)
    assert (0 > delta > -num_edges_per_new_node)

    # Adding edges in one go

    rng = np.random.default_rng(seed=seed)
    unifs = rng.uniform(0, 1, (num_nodes - 1) * num_edges_per_new_node)

    # numba
    if not polya_flag:

        edge_list_dummy = pa_generator_numba(
            10, 3, -1, unifs[:27])  # let numba warm up
        edge_list = pa_generator_numba(
            num_nodes, num_edges_per_new_node, delta, unifs)

    # polya
    else:

        b = (2 * num_edges_per_new_node + delta) * \
            np.arange(num_nodes) - num_edges_per_new_node
        b[0] = 1
        betas = rng.beta(
            num_edges_per_new_node + delta,
            b
        )
        betas[0] = 1

        edge_list = pa_generator_polya_numba(num_nodes, num_edges_per_new_node, delta,
                                             betas, unifs, sorted_flag, early_exit=early_exit)

    if python_object_flag:
        graph = ig.Graph()
        if early_exit is not None:
            graph.add_vertices(early_exit)
        else:
            graph.add_vertices(num_nodes)
        graph.add_edges(edge_list)

        return graph
    else:

        return edge_list


@jit(nopython=True)
def pa_generator_numba(num_nodes, num_edges_per_new_node, delta, rand_float):
    """
    Helper function for pa_generator
    sample the graph by definition

    """
    # Drawing an outcome from a distribution with pmf p0, ..., pk is the same as drawing a real
    # number from the uniform distribution on [0, 1] and find the first index such that the
    # cumulative sum of te pmf exceeds the number. To see this, think of throwing a dart on [0, 1]
    # with partition [0, p0], [p0, p0 + p1], [p0 + p1, p0 + p1 + p2], ...
    # Then the length of each interval is given by the pmf.

    # initialize graph with all vertices with (num_edges_per_new_node) edges
    # from each node (except node 0) pointing to node 0

    num_edges = (num_nodes - 1) * num_edges_per_new_node
    edge_list = [
        [0, 1 + int(np.floor(ei / num_edges_per_new_node))]
        for ei in range(num_edges)]

    # We maintain the cumulative sum of degree + delta
    cum_shifted_degs = (num_edges_per_new_node + delta) * np.ones(num_nodes)

    child = 1  # index of the newly added node

    # We add subsequent edges one by one.
    for ei in range(num_edges_per_new_node, num_edges):

        if ei % num_edges_per_new_node == 0:
            cum_shifted_degs[child] += cum_shifted_degs[child - 1]
            child += 1

        # choose the parent node for the edge
        # choose_parent samples from a distribution by
        # comparing a random real number with the non-normalized cumulative
        # distribution
        parent = choose_parent(cum_shifted_degs[:child], rand_float[ei])
        # print(ei, child, parent, cum_shifted_degs[:child])

        # update the graph and the degree sequence
        edge_list[ei][0] = parent
        cum_shifted_degs[parent:child] += 1

    # for e in edge_list: print(e)
    return edge_list


@jit(nopython=True)
def choose_parent(cum_shifted_degs, rand_float):

    # Given the strictly increasing sequence cum_shfited_degs
    # find the index of the first entry that exceeds rand_float * max( cum_shifted_degs)
    # use bisection method

    rand_float *= cum_shifted_degs[-1]

    if rand_float <= cum_shifted_degs[0]:
        return 0

    lower = 0
    upper = len(cum_shifted_degs) - 1

    while upper - lower > 1:
        trial_idx = int((upper + lower)/2)
        if rand_float <= cum_shifted_degs[trial_idx]:
            upper = trial_idx
        else:
            lower = trial_idx

    return upper


@jit(nopython=True)
def pa_generator_polya_numba(num_nodes, num_edges_per_new_node, delta, betas, unifs, sorted_flag, early_exit=None):
    """
    sample the graph by the Polya urn model
    """

    # betas is a list of num_nodes independent random variables with
    # betas[i] ~ Beta(
    #    num_edges_per_new_node + delta,
    #    num_edges_per_new_node * (2i - 1)) + delta * i
    # )
    # unifs is a list of independent uniformly distributed rv on [0, 1].

    if early_exit is None:
        early_exit = num_nodes

    m = num_edges_per_new_node
    num_edges = (early_exit - 1) * m

    if sorted_flag:
        for i in range(num_nodes - 1):
            unifs[(i*m):((i+1)*m)] = np.sort(unifs[(i*m):((i+1)*m)])

    phis = np.empty(num_nodes)
    phis[-1] = betas[-1]
    for i in range(num_nodes-1, 0, -1):
        phis[i-1] = phis[i] * betas[i-1] * (1 - betas[i]) / betas[i]

    Ss = np.empty(num_nodes)
    Ss[0] = phis[0]

    edge_list = [
        [0, 1 + int(ei // num_edges_per_new_node)]
        for ei in range(num_edges)]

    child = 1  # index of the newly added node

    for ei in range(num_edges_per_new_node, num_edges):

        if ei % num_edges_per_new_node == 0:
            Ss[child] = Ss[child-1] + phis[child]
            child += 1

        edge_list[ei][0] = choose_parent(Ss[:child], unifs[ei])

    return edge_list
