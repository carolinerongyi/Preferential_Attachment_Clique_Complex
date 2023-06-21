#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:22:10 2023

@author: alexsiu
"""
import numpy as np
import igraph as ig
from ripser import ripser
from persim import plot_diagrams
from numba import jit
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/CarolineHerr/Documents/GitHub/Preferential_Attachment_Clique_Complex')
from simulator_pa import pa_generator
import betti

import matplotlib.pyplot as plt

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
    #while start < len(edge_list) and end <= len(edge_list):

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
                if j == np.inf: betti_nums[dim, current_node] += 1

        current_node += 1
        start += m
        end += m

    return betti_nums

@jit(nopython=True)
def first_summand_lower_bound(edge_list, num_nodes, m, square, latest_node_in_square, numba_flag=True):

    first_summand = np.zeros(num_nodes, dtype = int)
    found_one_flag = 0
    for i in range(latest_node_in_square + 1, num_nodes):
        if True:
            fill = betti.check_node_square_connection(
                edge_list, i, square, m = m)
        if fill:
            if found_one_flag == 0:  # The first filling does not count
                found_one_flag = 1
            else:
                first_summand[i] = 1

    return first_summand

def second_summand_lower_bound(edge_list, num_nodes, square, first_summand):
    
    latest_node_in_square = max(square)
    second_summand = np.zeros(num_nodes, dtype = int)
    for i in range(latest_node_in_square + 1, num_nodes):
        if first_summand[i] >= 1:
            mat = betti.get_link_matrix(graph, square, i)
            dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
            second_summand[i] = int(
                any([pt[1] == 2 for pt in dgms[1]])
            )
    return second_summand

def betti2_lower_bound(graph, time = 20, first_summand = None, second_summand = None, links_betti_nums = None):

    """
    Return the lower bound of Betti 2. 
    The lower bound is found by the above inequality.
    
    OUTPUT
    Betti 2 lower bound at each time step as a list
    
    INPUT
    graph: igraph object
    time: the time constraint for the appearance of the square
    """
    
    print('running')
    
    edge_list = np.array([e.tuple for e in graph.es])
    num_nodes = graph.vcount()
    m = int(len(edge_list)/(num_nodes - 1))
    
    # find if there's a square in the first 20 nodes
    boo, square = betti.check_square_appearance(graph, time)
    
    print('found square')
    
    if not boo:
        first_term = np.zeros(num_nodes, dtype = int)
    else:
        latest_node_in_square = max(square)
        if first_summand is None:
            first_summand = first_summand_lower_bound(edge_list, num_nodes, m, square, latest_node_in_square)
        first_term = np.cumsum(first_summand)
    
    print('done first term')
    
    if not boo:
        second_term = np.zeros(num_nodes, dtype = int)
    else:
        if second_summand is None:
            second_summand = second_summand_lower_bound(edge_list, num_nodes, square, first_summand)
        second_term = np.cumsum(second_summand)

    if links_betti_nums is None:
        links_betti_nums = betti_numbers_of_links(graph, num_nodes, m, maxdim = 2)
    
    third_term = np.cumsum(links_betti_nums[2, :])
    
    return first_term - second_term - third_term, first_summand, second_summand, links_betti_nums

m = 7
delta = -5
num_nodes = 500
seed = 100
graph = pa_generator(num_nodes, m, delta, seed)

mat = betti.get_age_matrix(graph) # obtain the age matrix of the complex as an input for ripser
dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms'] # get the persistence diagrams
betti2_actual = betti.translate_PD_to_betti(dgms[2], num_nodes) # read the betti numbers from persistence diagrams


links_betti_nums = betti_numbers_of_links(graph, num_nodes, m, maxdim = 2)
betti2_upper = np.cumsum(links_betti_nums[1, :])

fig, ax = plt.subplots()
ax.plot(betti2_actual, label = 'actual')
ax.plot(betti2_upper, label = 'upper')

betti2_lower, _, _, _ = betti2_lower_bound(graph, num_nodes, links_betti_nums = links_betti_nums)