#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:22:10 2023

@author: alexsiu
"""
import betti
from simulator_pa import pa_generator
import numpy as np
import igraph as ig
from ripser import ripser
from persim import plot_diagrams
from numba import jit
import matplotlib.pyplot as plt
import sys
from itertools import combinations
sys.path.insert(
    0, '/Users/CarolineHerr/Documents/GitHub/Preferential_Attachment_Clique_Complex')


def check_square_appearance(graph, time):
    if type(time) == int:
        total_nodes = [i for i in range(time)]
        # try to pick the square with the least sum
        all_combinations = sorted(
            [[a, b, c, d] for a, b, c, d in combinations(total_nodes, 4)], key=sum)
    else:
        all_combinations = [time]
    for combination in all_combinations:
        subgraph = graph.induced_subgraph(combination)
        mat = betti.get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
        if len(dgms[1] == 1):
            return (True, combination)
    return (False, None)


def betti_numbers_of_links(graph, num_nodes, m, maxdim):
    """
    Return the upper bound for Betti 2 by finding the cumulative sum of Betti 1 of each node's link

    OUTPUT
    betti 2 upper bound at each time step as a list

    INPUT
    graph: igraph object

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
        mat = betti.get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=maxdim)['dgms']

        for dim in range(maxdim):
            for k, j in dgms[dim]:
                if j == np.inf:
                    betti_nums[dim, current_node] += 1

        current_node += 1
        start += m
        end += m

    return betti_nums


def first_summand_lower_bound(edge_list, num_nodes, m, square, latest_node_in_square):

    first_summand = np.zeros(num_nodes, dtype=int)
    found_one_flag = 0
    for i in range(latest_node_in_square + 1, num_nodes):
        if True:
            fill = betti.check_node_square_connection(
                edge_list, m, i, square)
        if fill:
            if found_one_flag == 0:  # The first filling does not count
                found_one_flag = 1
            else:
                first_summand[i] = 1

    return first_summand


def get_first_term_in_lower_bound(edge_list, square, largest_square_node, t):
    first_term = []
    for idx in range(largest_square_node+1):
        first_term.append(int(0))

    found_one_flag = 0
    for i in range(largest_square_node+1, t):
        if True:
            fill = betti.check_node_square_connection(
                edge_list, m, i, square)
        # else:
        #     fill = check_node_square_connection(graph, i, square)
        if fill:
            if found_one_flag == 0:  # The first filling does not count
                first_term.append(0)
                found_one_flag = 1
            else:
                first_term.append(1)
        else:
            first_term.append(0)

    return first_term


def second_summand_lower_bound(graph, num_nodes, square, first_summand):

    latest_node_in_square = max(square)
    second_summand = np.zeros(num_nodes, dtype=int)
    for i in range(latest_node_in_square + 1, num_nodes):
        if first_summand[i] >= 1:
            mat = betti.get_link_matrix(graph, square, i)
            dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
            second_summand[i] = int(
                any([pt[1] == 2 for pt in dgms[1]])
            )
    return second_summand


def lower_bound_link_relative_betti(graph, square, first_term):
    t = graph.vcount()
    edge_list = np.array([e.tuple for e in graph.es])
    # m = int(len(edge_list)/(edge_list[-1][1]))
    largest_square_node = max(square)
    link_betti2 = [0 for idx in range(largest_square_node+1)]
    for i in range(largest_square_node + 1, t):
        if first_term[i] >= 1:
            mat = betti.get_link_age_matrix(graph, square, i)
            dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
            link_betti2.append(int(
                any([pt[1] == 2 for pt in dgms[1]])
            ))
        else:
            link_betti2.append(0)  # Chunyin: I think we need this.
    return np.cumsum(link_betti2)


def betti2_lower_bound(graph, time=20, first_summand=None, second_summand=None, links_betti_nums=None):
    """
    Return the lower bound of Betti 2. 
    The lower bound is found by the above inequality.

    OUTPUT
    Betti 2 lower bound at each time step as a list

    INPUT
    graph: igraph object
    time: the time constraint for the appearance of the square
    """

    # print('running')

    edge_list = np.array([e.tuple for e in graph.es])
    num_nodes = graph.vcount()
    m = int(len(edge_list)/(num_nodes - 1))

    # find if there's a square in the first 20 nodes
    boo, square = betti.check_square_appearance(graph, time)

    # print('found square')

    if not boo:
        first_term = np.zeros(num_nodes, dtype=int)
    else:
        latest_node_in_square = max(square)
        if first_summand is None:
            first_summand = first_summand_lower_bound(
                edge_list, num_nodes, m, square, latest_node_in_square)
        first_term = np.cumsum(first_summand)

    # print('done first term')

    if not boo:
        second_term = np.zeros(num_nodes, dtype=int)
    else:
        if second_summand is None:
            second_summand = second_summand_lower_bound(
                edge_list, num_nodes, square, first_summand)
        second_term = np.cumsum(second_summand)

    if links_betti_nums is None:
        links_betti_nums = betti_numbers_of_links(
            graph, num_nodes, m, maxdim=2)

    third_term = np.cumsum(links_betti_nums[2, :])

    return first_term - second_term - third_term, first_summand, second_summand, links_betti_nums


m = 7
delta = -5
num_nodes = 10000
graph1 = pa_generator(num_nodes, m, delta, 100)
graph2 = pa_generator(num_nodes, m, delta, 130)
graph3 = pa_generator(num_nodes, m, delta, 156)
graph4 = pa_generator(num_nodes, m, delta, 53)

# graph2 = ig.Graph()
# graph2.add_vertices(4)
# graph2.add_edges([(0, 1), (1, 2), (2, 3), (0, 3)])

# graph3 = ig.Graph()
# graph3.add_vertices(5)
# graph3.add_edges([(0, 1), (1, 2), (2, 3), (0, 3),
#                  (0, 4), (1, 4), (2, 4), (3, 4)])


# graph4 = ig.Graph()
# graph4.add_vertices(6)
# graph4.add_edges([(0, 1), (1, 2), (2, 3), (0, 3),
#                  (0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5)])

# obtain the age matrix of the complex as an input for ripser
# mat = betti.get_age_matrix(graph)
# dgms = ripser(mat, distance_matrix=True, maxdim=2)[
#     'dgms']  # get the persistence diagrams
# read the betti numbers from persistence diagrams
# betti2_actual = betti.translate_PD_to_betti(dgms[2], num_nodes)
# s = [graph1, graph2, graph3, graph4]

# for graph in s:
# for seed in np.arange(100):
#     graph = pa_generator(num_nodes, m, delta, seed)
#     # edge_list = np.array([e.tuple for e in graph.es])
#     # square = betti.check_square_appearance(graph, 20)[1]

#     a = betti2_lower_bound(graph, time=20)[0]
#     b = betti.betti2_lower_bound(graph, time=20)[0]

#     if sum(a + b) > 0:
#         print((a == b).all())  # do not print if both lower bounds are 0

m = 7
T0 = 60
seed = 100
graph1 = pa_generator(T0, m, -6.5, seed)
graph2 = pa_generator(T0, m, -0.5, seed)
layout = graph.layout_circle()
labels = [x for x in range(T0)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ig.plot(graph1, target = ax1, layout=layout, vertex_label = labels, edge_color = '#36454F')
ig.plot(graph2, target = ax2, layout=layout, vertex_label = labels, edge_color = '#36454F')
