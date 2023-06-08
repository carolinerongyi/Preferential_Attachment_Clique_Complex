#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:54:15 2022

@author: CarolineHerr
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import time
from datetime import datetime
from tqdm import tqdm

from ripser import ripser
from persim import plot_diagrams
import simulator_pa as pa
from itertools import combinations

import matplotlib.pyplot as plt
import igraph as ig
import os
import time
import powerlaw
import csv
from sklearn.linear_model import LinearRegression
from numba import jit
from numba.typed import List
from collections import Counter
# import caffeine

NNN = 40000


def get_age_matrix(graph):
    num_nodes = graph.vcount()
    dm = np.zeros((num_nodes, num_nodes))
    edge_list = graph.es
    for edge in edge_list:
        i, j = edge.tuple[0], edge.tuple[1]
        dm[i, j] = max(i, j)

    for i in range(num_nodes):
        dm[i][i] = i
    return dm


def num_self_loops(graph):
    g = graph.copy().simplify(multiple=False, loops=True)
    return graph.ecount() - g.ecount()


def log_log_plot(num_nodes, num_edges_per_new_node, delta, seed):
    g = pa.pa_generator(num_nodes, num_edges_per_new_node, delta, seed)
    deg = g.degree()
    x, y = np.unique(deg, return_counts=True)
    y = np.array(y)/len(deg)
    x = x[:len(x)//4]
    y = y[:len(y)//4:]
    reg = LinearRegression()
    reg.fit(np.log(x).reshape((-1, 1)), np.log(y).reshape((-1, 1)))
    coef = reg.coef_[0, 0]
    plt.loglog(x, y, '.')
    return coef

# return a list of all repeated edge list sorted


def repeated_edge_list(graph):
    def getKey(item):
        return item[1]
    edge_list = [e.tuple for e in graph.es]
    sorted_edge_list = sorted(edge_list, key=getKey)
    repeated_edge_list = []
    for i in range(1, len(sorted_edge_list)):
        if (sorted_edge_list[i][0] == sorted_edge_list[i-1][0]) and \
                (sorted_edge_list[i][1] == sorted_edge_list[i-1][1]):
            repeated_edge_list.append(sorted_edge_list[i])
    return repeated_edge_list


def num_multiples(repeated_edge_list, num_edge, t):
    num_repeated_edges = len(repeated_edge_list)
    if t == 0:
        num_repeated_edges = 0
    for i in range(len(repeated_edge_list)):
        if repeated_edge_list[i][1] > t:
            num_repeated_edges = i
            break
    if t == 0:
        proportion = 0
    else:
        proportion = num_repeated_edges/(t*num_edge)
    return num_repeated_edges, proportion


def find_exponent(y):
    fit = powerlaw.Fit(y, discrete=True)
    return fit.power_law.alpha


def num_triangles(graph):
    # graph = graph.simplify()
    triangle = ig.Graph(directed=False)
    triangle.add_vertices(3)
    triangle.add_edges([
        (1, 0),
        (2, 0),
        (2, 1)
    ])
    triangle_count = graph.get_subisomorphisms_vf2(triangle)
    triangle_count = sorted([sorted(i) for i in triangle_count])
    triangle_count = [triangle_count[i] for i in range(
        len(triangle_count)) if i == 0 or triangle_count[i] != triangle_count[i-1]]
    return triangle_count


@jit(nopython=True)
def custom_append(L, a, n):
    for i in range(n):
        L.append(a)
    return(L)


@jit(nopython=True)
def custom_cumsum(L):

    cumsumL = List()
    cumsumL.append(L[0])
    [cumsumL.append(cumsumL[i] + L[i+1]) for i in range(len(L)-1)]

    return cumsumL

# use counter to return a list of lists. Each element contains [node, node, multiplicity]
# for each pair of edge


def count_multiplicity(edge_list):
    count = Counter(edge_list)
    edge_list_with_multiplicity = []
    node_location = [0]
    current = 0
    for i in count.keys():
        node1 = i[0]
        node2 = i[1]
        multiplicity = count[i]
        edge_list_with_multiplicity.append([node1, node2, multiplicity])
        if i[1] != current:
            current = i[1]
            node_location.append(len(edge_list_with_multiplicity) - 1)
    node_location.append(len(edge_list_with_multiplicity))
    return np.array(edge_list_with_multiplicity), np.array(node_location)


@jit(nopython=True)
def count_multiplicity_numba(edge_list):
    # we need the edge list to be sorted!!!
    node_location = [0]
    edge_list_with_multiplicity = []
    previous_edge = (-1, -1)
    num_simple_edges = 0
    for e in edge_list:
        if e != previous_edge:
            edge_list_with_multiplicity.append([e[0], e[1], 1])
            if e[1] != previous_edge[1]:  # 1 if we have moved to a new child
                node_location.append(num_simple_edges)
            num_simple_edges += 1
        else:
            edge_list_with_multiplicity[-1][2] += 1
        previous_edge = e
    node_location.append(len(edge_list_with_multiplicity))

    return np.array(edge_list_with_multiplicity), np.array(node_location)


def test_count_multiplicity():
    graph = pa.pa_generator(500, 3, -2, 3, polya_flag=1, sorted_flag=1)
    edge_list = [e.tuple for e in graph.es]
    mlist_numba, nlist_numba = count_multiplicity_numba(edge_list)
    mlist, nlist = count_multiplicity(edge_list)

    # print(edge_list)

    # print('numba')
    # print(mlist_numba)
    # print(nlist_numba)

    # print('old')
    # print(mlist)
    # print(nlist)
    m = mlist == mlist_numba
    n = nlist == nlist_numba
    print(m.all())
    print(n.all())

# test_count_multiplicity()


@jit(nopython=True)
def num_triangles_over_time(edge_list, edge_list_with_multiplicity, node_location):
    start = 0
    num_edges_per_new_node = int(len(edge_list)/(edge_list[-1][1]))
    end = num_edges_per_new_node
    new_tri_in_batch = List()
    new_tri_in_batch.append(0)
    # num_nodes = edge_list[-1][1] + 1
    # multiplicity_matrix = np.zeros(shape = (num_nodes, num_nodes))
    # print('    building multiplicity matrix')
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         max_index = num_edges_per_new_node * j + 1
    #         min_index = num_edges_per_new_node * (j-1)
    #         multiplicity_matrix[i,j] = edge_list[min_index:max_index].count((i,j))

    while (start < len(edge_list) and end <= len(edge_list)):
        parents_list = List()
        [parents_list.append(edge_list[i][0]) for i in range(start, end)]
        count = 0
        for ii in range(len(parents_list)-1):
            for j in parents_list[ii+1:]:
                node1 = min(parents_list[ii], j)
                node2 = max(parents_list[ii], j)

                # locate the edge in the multiplicity list
                s = node_location[node2]
                e = node_location[node2+1]
                for loc in range(s, e):
                    if edge_list_with_multiplicity[loc][0] == node1:
                        count += edge_list_with_multiplicity[loc][2]

                # count += int(multiplicity_matrix[node1, node2])
        new_tri_in_batch.append(count)
        start += num_edges_per_new_node
        end += num_edges_per_new_node
    return custom_cumsum(new_tri_in_batch)


# t = 10000
# num_graphs = 1000
# for i in range(num_graphs):
#     graph = pa.pa_generator(t, 7, -5, 3*i, polya_flag=1, sorted_flag=1)
#     typed_edge_list = List()
#     [typed_edge_list.append(e.tuple) for e in graph.es]
#     edge_list_with_multiplicity, node_location = count_multiplicity_numba(
#         typed_edge_list)
#     tri = num_triangles_over_time_v2(
#         typed_edge_list, edge_list_with_multiplicity, node_location)
#     plt.plot(np.array(tri)/np.arange(1, t+1))
# plt.show()
# graph = pa.pa_generator(5,3,-2,0)
# edge_list = [(e.tuple) for e in graph.es]
# multiplicity, node_loc = count_multiplicity(edge_list)
# print(num_triangles_over_time_v2(edge_list, multiplicity, node_loc))


def num_squares(g):
    g = g.simplify()
    square = ig.Graph(directed=False)
    square.add_vertices(4)
    square.add_edges([
        (1, 0),
        (2, 0),
        (3, 1),
        (3, 2)
    ])
    square_count = g.get_subisomorphisms_lad(square)
    square_count = sorted([sorted(i) for i in square_count])
    square_count = [square_count[i] for i in range(
        len(square_count)) if i == 0 or square_count[i] != square_count[i-1]]
    return square_count


def num_square_capstone(graph):
    graph = graph.simplify()
    square_capstone = ig.Graph()
    square_capstone.add_vertices(5)
    square_capstone.add_edges([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4)
    ])
    count = graph.get_subisomorphisms_vf2(square_capstone)
    count = sorted([sorted(i) for i in count])
    count = [count[i]
             for i in range(len(count)) if i == 0 or count[i] != count[i-1]]
    return count


def num_square_capstone_v2(num_nodes, num_edges_per_new_node, delta, seed):
    graph = pa.pa_generator(num_nodes, num_edges_per_new_node, delta, seed)
    edge_list = [e.tuple for e in graph.es]
    start = 0
    end = num_edges_per_new_node
    new_capstone_in_batch = [0]
    while start < len(edge_list) and end <= len(edge_list):
        batch = set()  # set of parents
        for i in range(start, end):
            if edge_list[i][0] not in batch:
                batch.add(edge_list[i][0])
        subgraph = graph.induced_subgraph(list(batch))
        new_capstone_in_batch.append(len(num_squares(subgraph)))
        start += num_edges_per_new_node
        end += num_edges_per_new_node
    return np.cumsum(new_capstone_in_batch)


def betti_num_evolution(num_nodes, num_edges_per_new_node, delta, num_graphs, dimension=2):
    """
    dimension should be a list of betti number dimensions
    return betti 1, 2

    """
    t0 = time.time()
    # plt.figure(0)
    diagrams = []

    graph_time = 0
    ripser_time = 0
    for i in range(num_graphs):
        graph = pa.pa_generator(num_nodes, num_edges_per_new_node, delta, i**3)
        graph_time += (time.time() - t0)
        t0 = time.time()
        mat = get_age_matrix(graph)
        dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']
        ripser_time += (time.time() - t0)
        t0 = time.time()
        diagrams.append(dgms)
    graph_time = graph_time/num_graphs
    ripser_time = ripser_time/num_graphs
    betti_2 = []
    dim = 2
    points = []
    for i in range(num_graphs):
        points.append(diagrams[i][dim])
        betti_num = np.zeros(num_nodes)
        for i, j in diagrams[i][dim]:
            betti_num[int(i):] += 1
            if j < float('inf'):
                betti_num[int(j):] -= 1
        plt.plot(betti_num, alpha=0.3, color='b')
        betti_2.append(betti_num)
    plt.show()
    return betti_2, graph_time, ripser_time


# not using. Use translate_PD_to_betti instead
def betti_2_evolution(num_nodes, num_edges_per_new_node, delta, num_graphs):
    """
    dimension should be a list of betti number dimensions
    return betti 1, 2

    """
    t0 = time.time()
    # plt.figure(0)
    diagrams = []

    for i in range(num_graphs):
        graph = pa.pa_generator(num_nodes, num_edges_per_new_node,
                                delta, i**3).simplify()
        mat = get_age_matrix(graph)
        dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']
        diagrams.append(dgms)
    betti_2 = []
    # plt.subplot(1,len(dimension),dim)
    for i in range(num_graphs):
        betti_num = 0
        for k, j in diagrams[i][2]:
            betti_num += 1
            if j < float('inf'):
                betti_num -= 1
        # plt.plot(betti_num, alpha = 0.3, color = 'b')
        betti_2.append(betti_num)
        # plt.title('Betti ' + str(dim))
    return betti_2


@jit(nopython=True)
def translate_PD_to_betti(diagram, max_fil):

    # diagram is a persistence diagram of a fixed dimension
    # max_fil is the maximum filtration value in the nested sequence
    # of spaces. In the case of the preferential attachment complex
    # It is T, the total number of nodes.

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


def loglog_betti_num(list_of_betti_nums, num_nodes, tau):
    plt.figure(1)
    y = np.array(range(num_nodes))**(1 - 4*(1 - 1/(tau - 1)))
    plt.plot(y)
    for i in range(len(list_of_betti_nums)):
        plt.loglog(list_of_betti_nums[i])
    plt.show()


# num_nodes = 10000
# list_of_betti_nums = []
# for k in range(1000):
#     graph = pa.pa_generator(num_nodes, 7, -5, k**3).simplify()
#     mat = get_age_matrix(graph)
#     dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']
#     betti_nums = translate_PD_to_betti(dgms[2], num_nodes)
#     list_of_betti_nums.append(betti_nums)
#     plt.plot(betti_nums, alpha=0.3, color='b')
# plt.xlabel('Time')
# plt.ylabel('Number of Betti 2')
# plt.savefig('betti2.png', dpi=500)
# plt.show()


# this function only returns one betti_0, which is correct
def testing_ripser_simplify():
    graph = ig.Graph()
    graph.add_vertices(2)
    graph.add_edges([(0, 1), (0, 1)])
    mat = get_age_matrix(graph)
    dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']
    plot_diagrams(dgms, show=True)


# plot number of capstones vs the number of births
def capstone_v_births(num_nodes, num_edges_per_new_node, delta, seed):
    graph = pa.pa_generator(num_nodes, num_edges_per_new_node, delta, seed)
    mat = get_age_matrix(graph)
    points = ripser(mat, distance_matrix=True, maxdim=2)['dgms'][2]
    betti2 = np.zeros(num_nodes)
    for i, j in points:
        betti2[int(i):] += 1
    capstone = num_square_capstone_v2(
        num_nodes, num_edges_per_new_node, delta, seed)
    plt.plot(capstone, betti2, alpha=0.3, color='b')

# create and save "num_graphs" number of graphs that have "num_nodes" number of nodes
# num_edges and delta


def create_graphs_pickle(num_graphs, num_nodes, num_edges, delta):
    t0 = time.time()
    for i in range(num_graphs):
        g = pa.pa_generator(num_nodes, num_edges, delta, i**3)
        name = 'graph_collections/' + str(num_nodes) + ' nodes ' + \
            str(num_edges) + ' edges ' + \
            str(delta) + ' delta seed ' + str(i**3)
        g.write_pickle(fname=name)
    print(time.time() - t0)
    return


def create_graphs_csv(num_graphs, num_nodes, num_edges, delta):
    t0 = time.time()
    for i in range(num_graphs):
        g = pa.pa_generator(num_nodes, num_edges, delta, i**3)
        print(time.time() - t0)
        edge_list = [list(e.tuple) for e in g.es]
        filename = 'graph_collections/' + str(num_edges) + '_edges_' + \
            str(-delta) + '_delta_' + str(num_nodes) + \
            '_nodes_' + str(i) + '.csv'
        file = open(filename, 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(edge_list)


def read_csv_file(filename):
    edge_list = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            edge_list.append([int(i) for i in row])
    graph = ig.Graph()
    num_nodes = edge_list[-1][1] + 1
    graph.add_vertices(num_nodes)
    graph.add_edges(edge_list)
    return graph


def tri_exponent(m, delta):
    tau = 3 + delta/m
    return (3 - tau)/(tau - 1)


def subgraph_lemma_max_difference(num_nodes, num_edges_per_new_node, delta, seed):
    graph = pa.pa_generator(num_nodes, num_edges_per_new_node, delta, seed)
    max_difference = 0
    cumulative_degree = num_edges_per_new_node + delta
    for i in range(1, num_nodes):
        cumulative_degree += graph.degree(i) + delta
        difference = abs((i/num_nodes)**((i + delta)/(2*num_edges_per_new_node + delta))
                         - cumulative_degree)
        max_difference = max(max_difference, difference)
    return max_difference


def num_tetrahedra_over_time(graph):
    edge_list = [e.tuple for e in graph.es]
    start = 0
    num_edges_per_new_node = int(graph.ecount()/(graph.vcount() - 1))
    end = num_edges_per_new_node
    new_tetra_in_batch = [0]
    i = 0
    while start < len(edge_list) and end <= len(edge_list):
        batch = set()  # set of parents
        for i in range(start, end):
            if edge_list[i][0] not in batch:
                batch.add(edge_list[i][0])
        subgraph = graph.induced_subgraph(list(batch))
        new_tetra_in_batch.append(len(num_triangles(subgraph)))
        start += num_edges_per_new_node
        end += num_edges_per_new_node
    return np.cumsum(new_tetra_in_batch)


def expectation_tri(m, delta, t):
    tau = 3 + delta/m
    return (((m**2)*(m-1)*(m+delta)*(m+delta+1))/((delta**2)*(2*m + delta)))*(t**((3-tau)/(tau-1)))*np.log(t)*(1+1)

# input: m = number of edges per new node; delta = delta
# t = the size of graph; n = number of graphs to create
# empiricial_or_not: whether the expectation is empirical or theoretical (boolean)
# plot a histogram and return the mean of N/E(N)


def expectation_test(m, delta, t, n, empirical_or_not, directory, filename, ax=None):
    res = []
    total = 0
    count_list = []

    print('warming up')
    # This part do all expensive computations for a small example to help
    # numba warm up. Results here will not be used later on.
    dummy_graph = pa.pa_generator(50, 4, -2, 3, polya_flag=1, sorted_flag=1)
    typed_edge_list = List()
    [typed_edge_list.append(e.tuple) for e in dummy_graph.es]
    # edge_list_with_multiplicity, node_location = count_multiplicity(typed_edge_list)
    edge_list_with_multiplicity, node_location = count_multiplicity_numba(
        typed_edge_list)
    tri = num_triangles_over_time(
        typed_edge_list, edge_list_with_multiplicity, node_location)

    for i in tqdm(range(n)):

        print('generating graph')
        graph = pa.pa_generator(t, m, delta, 3*i, polya_flag=1, sorted_flag=1)
        # edge_list = [e.tuple for e in graph.es]
        # tri = num_triangles_over_time_v2(edge_list)

        print('expressing graph as edge_list')
        typed_edge_list = List()
        [typed_edge_list.append(e.tuple) for e in graph.es]
        # edge_list_with_multiplicity, node_location = count_multiplicity(typed_edge_list)
        edge_list_with_multiplicity, node_location = count_multiplicity_numba(
            typed_edge_list)

        print('counting triangle')
        tri = num_triangles_over_time(
            typed_edge_list, edge_list_with_multiplicity, node_location)

        count = tri[-1]
        total += count
        count_list.append(count)

        print('recording count')
        with open(os.path.join(directory, filename + '.txt'), 'a') as fp:
            fp.writelines(str(count))
            fp.writelines('\n')
        if i % int(np.ceil((n/10))) == 0:
            with open(os.path.join(directory, filename + '.log'), 'a') as fp:
                fp.writelines(
                    str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")))
                fp.writelines('    ' + filename + '    ' + str(i) + '\n')
    if empirical_or_not:
        expectation = total/n
    else:
        expectation = expectation_tri(m, delta, t)
    res = np.array(count_list)/expectation
    ax.hist(res)
    if empirical_or_not:
        emp_ext = " empirical"
    else:
        emp_ext = " theoretical"
    ax.set_title(filename + emp_ext)
    print(np.mean(res))
    return ax, list(count_list)

# save as csv instead of txt
# plot histogram of the final betti number


def save_betti2_csv():
    num_nodes = 50000
    num_edges_per_new_node = 8
    delta = -5
    num_graphs = 5
    betti2, graph_time, ripser_time = betti_num_evolution(
        num_nodes, num_edges_per_new_node, delta, num_graphs)
    print('graph_time ' + str(graph_time))
    print('ripser_time ' + str(ripser_time))
    betti2 = np.array(betti2)[:, -1]
    plt.hist(betti2)
    plt.show()
    filename = 'data/betti2_count/' + str(num_edges_per_new_node) + '_' + str(-delta) + '_' + str(
        num_nodes) + '_' + str(num_graphs) + '_betti2_sequence.csv'
    betti2 = [[str(x)] for x in betti2]
    with open(filename, 'w+', newline='') as f:
        write = csv.writer(f)
        write.writerows(betti2)


def num_triangles_evolution_scaled(t, seed):
    m = 7
    delta = -5
    graph = pa.pa_generator(t, m, delta, seed, polya_flag=1, sorted_flag=1)
    typed_edge_list = List()
    [typed_edge_list.append(e.tuple) for e in graph.es]
    edge_list_with_multiplicity, node_location = count_multiplicity_numba(
        typed_edge_list)
    tri = num_triangles_over_time
    (
        typed_edge_list, edge_list_with_multiplicity, node_location)
    expected_tri = np.array([expectation_tri(m, delta, i) for i in range(t)])
    return tri/expected_tri


def betti2_upper_bound(graph):

    edge_list = np.array([e.tuple for e in graph.es])
    start = 0
    num_edges_per_new_node = int(len(edge_list)/(edge_list[-1][1]))
    end = num_edges_per_new_node
    new_betti1_in_batch = [0]
    new_betti2_in_batch = [0]

    while start < len(edge_list) and end <= len(edge_list):
        # find parents of each node
        parents_list = edge_list[start:end, 0]
        # parents_list = List()
        # [parents_list.append(edge_list[i][0]) for i in range(start, end)]
        parents_list = np.unique(parents_list)

        # generate subgraph for the parents
        subgraph = graph.induced_subgraph(parents_list)

        # count the betti numbers
        mat = get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']

        betti1 = 0
        for k, j in dgms[1]:
            betti1 += 1
            if j < float('inf'):
                betti1 -= 1

        betti2 = 0
        for k, j in dgms[2]:
            betti2 += 1
            if j < float('inf'):
                betti2 -= 1

        new_betti1_in_batch.append(betti1)
        new_betti2_in_batch.append(betti2)
        start += num_edges_per_new_node
        end += num_edges_per_new_node
    return np.cumsum(new_betti1_in_batch), np.cumsum(new_betti2_in_batch)


def verify_upper_bound(num_nodes, num_edges_per_new_node, delta, num_graphs):
    betti_2_actual = betti_2_evolution(
        num_nodes, num_edges_per_new_node, delta, num_graphs)  # uses seed i**3
    betti_2_upper_bound = []
    for i in range(num_graphs):
        betti_2_upper_bound.append(betti2_upper_bound(
            num_nodes, num_edges_per_new_node, delta, i**3)[-1])
    for i in range(num_graphs):
        if betti_2_upper_bound[i] < betti_2_actual[i]:
            print((betti_2_upper_bound[i], betti_2_actual[i], i))
            return False
    return True


# Plot histogram of betti_actual/betti_upperbound


# check if there's a square by time


######### lower bound #########
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
        mat = get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
        if len(dgms[1] == 1):
            return (True, combination)
    return (False, None)

# square is a list of nodes that form a square


@jit(nopython=True)
def check_node_square_connection(edge_list, node, square, m=7, par_col=0):
    node_parents = edge_list[(node - 1) * m:node * m, par_col]
    for s in square:
        if s not in node_parents:
            return False
    return True


def proportion_square_appearance(time):
    n = 1000
    times = 0
    for i in range(n):
        g = pa.pa_generator(time, 7, -5, i**3)
        exists, square = check_square_appearance(g, time)
        if exists:
            times += 1
    return times/1000


# problem: induced_subgraph mixes the index of the nodes
# assume node fills the square
def get_link_matrix(graph, square, node):
    link = graph.neighbors(node)
    link = [i for i in link if i <= node]
    # link.append(node) # I think the link should not contain node???
    num_nodes = len(link)
    dm = np.zeros((num_nodes, num_nodes))
    subg = graph.induced_subgraph(link)

    # find the index of square nodes in the induced subgraph
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


# assume there's a square in the first 20 nodes
def lower_bound_link_relative_betti(graph, square, first_term):
    t = graph.vcount()
    edge_list = np.array([e.tuple for e in graph.es])
    # m = int(len(edge_list)/(edge_list[-1][1]))
    largest_square_node = max(square)
    link_betti2 = [0 for idx in range(largest_square_node+1)]
    for i in range(largest_square_node + 1, t):
        if first_term[i] >= 1:
            mat = get_link_matrix(graph, square, i)
            dgms = ripser(mat, distance_matrix=True, maxdim=1)['dgms']
            # we only need the diagram at dimension 1

# --------------------------------------------------------
            link_betti2.append(int(
                any([pt[1] == 2 for pt in dgms[1]])
            ))
            # Chunyin: I suggest the codes above.
            # It does the same thing but there are fewer if loops.
            # If it doesn't make sense to you (or if it is wrong)
            # feel free to use your original codes

        #     for pt in dgms[1]:
        #         if pt[1] == 2:
        #             indicator = True
        # if indicator:
        #     link_betti2.append(1)
        #     indicator = False
        # else:
        #     link_betti2.append(0)
# --------------------------------------------------------
        else:
            link_betti2.append(0)  # Chunyin: I think we need this.
    return np.cumsum(link_betti2)


@jit(nopython=True)
def get_first_term_in_lower_bound(square, largest_square_node, t, numba_flag=True, graph=None, edge_list=None, m=None, par_col=None):
    first_term = List()
    for idx in range(largest_square_node+1):
        first_term.append(int(0))

    found_one_flag = 0
    for i in range(largest_square_node+1, t):
        if True:
            fill = check_node_square_connection(
                edge_list, i, square, m=m, par_col=par_col)
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


def betti2_lower_bound(graph, time=20, m=None):
    t = graph.vcount()
    boo, square = check_square_appearance(graph, time)

    edge_list = np.array([e.tuple for e in graph.es])
    num_edges_per_new_node = int(len(edge_list)/(edge_list[-1][1]))
    if not boo:
        first_term_sum = np.array([0 for idx in range(t)])
    else:
        largest_square_node = max(square)
        first_term = get_first_term_in_lower_bound(np.array(
            square), largest_square_node, t, edge_list=edge_list, m=num_edges_per_new_node, par_col=0)
        first_term_sum = np.cumsum(first_term)

    if not boo:
        second_term = np.array([0 for idx in range(t)])
    else:
        second_term = lower_bound_link_relative_betti(
            graph, square, first_term)

    start = 0
    end = num_edges_per_new_node
    new_betti2_in_batch = [0]

    while start < len(edge_list) and end <= len(edge_list):
        parents_list = edge_list[start:end, 0]
        parents_list = np.unique(parents_list)

        # generate subgraph for the parents
        subgraph = graph.induced_subgraph(parents_list)

        # count the betti numbers
        mat = get_age_matrix(subgraph)
        dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']

        betti2 = 0
        for k, j in dgms[2]:
            betti2 += 1
            if j < float('inf'):
                betti2 -= 1

        new_betti2_in_batch.append(betti2)
        start += num_edges_per_new_node
        end += num_edges_per_new_node
    third_term = np.cumsum(new_betti2_in_batch)
    lower_bound = first_term_sum - second_term - third_term
    return lower_bound, first_term_sum, second_term


def betti2_lower_terms_quick(t, m, delta, seed, time, **kwargs):

    if type(time) == int:
        early_exit = time
    else:
        early_exit = max(time)

    early_exit = 3*early_exit + 3

    graphlet = pa.pa_generator(
        t, m, delta, seed=seed, early_exit=early_exit, **kwargs)

    boo, square = check_square_appearance(graphlet, time)

    edge_list = np.array([e.tuple for e in graphlet.es])
    num_edges_per_new_node = int(len(edge_list)/(edge_list[-1][1]))

    if not boo:
        # [0]*t
        return [0 for idx in range(t)], [0 for idx in range(t)], [0 for idx in range(t)]
        # [0] * t is also correct but it is dangerous, because
        # if we define x = [0] * k
        # and want to change x[1] to 1
        # The whole array is changed!

    graph = pa.pa_generator(t, m, delta, seed=seed, **kwargs)
    edge_list = np.array([e.tuple for e in graph.es])
    largest_square_node = max(square)

    first_term = get_first_term_in_lower_bound(
        np.array(square), largest_square_node, t, edge_list=edge_list, m=m, par_col=0)
    first_term_sum = np.cumsum(first_term)

    # print(first_term_sum[-1])
    # if any(first_term_sum != first_term_test):
    #     print('no')

    # second term
    second_term = lower_bound_link_relative_betti(graph, square, first_term)

    difference = first_term_sum - second_term
    return difference, first_term_sum, second_term


def hist_ratio_betti2_actual_upperbound(n, m, delta, num_graphs, bound='upper', actual_flag=True, time=20, **kwargs):
    # ratio = []
    betti2_bound = np.empty(num_graphs, dtype=int)
    betti2_actual = np.empty(num_graphs, dtype=int)

    for i in tqdm(range(num_graphs)):
        graph = pa.pa_generator(n, m, delta, i**3, **kwargs)
        if bound == 'upper':
            betti_upper_tmp, betti_lower_tmp = betti2_upper_bound(graph)
            betti2_bound[i] = betti_upper_tmp[-1]
            # betti2_lower[i] = betti_lower_tmp[-1]
        else:
            betti_lower_tmp = betti2_lower_bound(graph, time)
            betti2_bound[i] = betti_lower_tmp[0][-1]

        if actual_flag:
            mat = get_age_matrix(graph)
            dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']
            betti2_actual[i] = translate_PD_to_betti(dgms[2], n)[-1]
        # ratio.append(betti_actual/betti_upper)
        # print(betti_lower)
    # plt.hist(ratio)
    # plt.show()
    return betti2_bound, betti2_actual

# hist_ratio_betti2_actual_upperbound(1000, 7, -5, 500)


def betti2_bound_sharpness_shell(n, m, delta, num_graphs, path, bound='upper', actual_flag=True, **kwargs):
    timestamp = str(datetime.now().strftime("%y%m%d_%H%M%S"))
    path += "_" + timestamp
    os.mkdir(path)
    os.chdir(path)
    with open('params.csv', 'w+', newline='') as g:
        g.write("n, {}\n".format(n))
        g.write("m, {}\n".format(m))
        g.write("delta, {}\n".format(delta))
        g.write("num_graphs, {}\n".format(num_graphs))
        for key, val in kwargs.items():
            g.write("{}, {}\n".format(key, val))
    betti2 = hist_ratio_betti2_actual_upperbound(
        n, m, delta, num_graphs, bound=bound, actual_flag=actual_flag, **kwargs)
    names = [bound, "actual"]
    for name, betti in zip(names, betti2):
        if name != "actual" or actual_flag:
            with open(name + '.csv', 'w+', newline='') as f:
                # save the data while generating it
                # so that we still have something if, say, we run out of memory
                write = csv.writer(f)
                write.writerow(betti)


# n = 10000
# m = 7
# delta = -5
# num_graphs = 1000
# polya_flag = True
# path = "data/betti2_bound_sharpness"

# bound = "lower"
# actual_flag = False

# # betti2_bound_sharpness_shell(n, m, delta, num_graphs, path, polya_flag = polya_flag)
# betti2_bound_sharpness_shell(
#     n, m, delta, num_graphs, path, polya_flag=polya_flag,
#     bound = bound, actual_flag = actual_flag
#     )

def plot_histo_upper(path, bound='upper', path_actual=None):
    if path_actual is None:
        path_actual = path

    actual = np.genfromtxt(os.path.join(
        path_actual, 'actual.csv'), delimiter=',')
    bound_array = np.genfromtxt(os.path.join(
        path, bound + '.csv'), delimiter=',')
    for A in bound_array:
        if A == 0:
            print('hi')
    if bound == 'upper':
        ratio = [A/U for A, U in zip(actual, bound_array)]
    else:
        ratio = [L/A for A, L in zip(actual, bound_array)]
    fig, ax = plt.subplots()
    ax.hist(ratio)
    ax.axvline(0, ls='--')
    ax.axvline(0.2, ls='--')


def plot_histo_actual(path):
    actual = np.genfromtxt(os.path.join(path, 'actual.csv'), delimiter=',')
    fig, ax = plt.subplots()
    ax.hist(actual)


def compress_lower_bound_file(file, new_file, starting_col=0):
    if file[-4:] == '.npy':
        array = np.load(file)
    elif file[-4:] == '.csv':
        array = np.genfromtxt(file, dtype=int, delimiter=',')

    array = array[:, starting_col:]
    new_array = array[:, 1:] - array[:, :-1]
    new_array = sparse.csc_array(new_array)
    sparse.save_npz(new_file, new_array)


def decompress_lower_bound_file(file):
    array = sparse.load_npz(file)
    new_array = np.cumsum(array.toarray(), axis=1)
    z = np.zeros(new_array.shape[0])[:, np.newaxis]
    return np.hstack((z, new_array))


# verify second term
# graph = ig.Graph()
# graph.add_vertices(6)
# graph.add_edges([(1, 0), (2, 1), (3, 2), (3, 0), (4, 0), (4, 1),
#                 (4, 3), (4, 2), (5, 0), (5, 1), (5, 2), (5, 3)])
# print(lower_bound_link_relative_betti(graph, [0, 1, 2, 3], [0, 0, 0, 0, 0, 1]))

# plot_histo_upper("data/betti2_bound_sharpness_230411_165631")
# plot_histo_actual("data/betti2_bound_sharpness_230411_165631")

# plot_histo_upper(
#     "data/betti2_bound_sharpness_230417_103458",
#     bound = 'lower',
#     path_actual = "data/betti2_bound_sharpness_230411_165631")
# g = pa.pa_generator(100, 7, -5, 729, polya_flag=1, sorted_flag=1)
# betti2_lower_bound(g)


# graph = pa.pa_generator(num_nodes, 7, -5, k**3).simplify()
#     mat = get_age_matrix(graph)
#     dgms = ripser(mat, distance_matrix=True, maxdim=2)['dgms']
#     betti_nums = translate_PD_to_betti(dgms[2], num_nodes)
#     list_of_betti_nums.append(betti_nums)s
#     plt.plot(betti_nums, alpha=0.3, color='b')


# with open(filename, "w") as output:
#     for row in betti2:
#         output.write(str(row) + '\n')
# expectation_test(4,-2,100,200, False, 'data', 'test', ax = plt.subplots()[1])


# LL = [4, 2, 5]
# L = List()
# [L.append(a) for a in LL]
# print(custom_cumsum(L))
# print(custom_cumsum(L)[-1])

# LL = [4, 2, 5]
# L = List()
# [L.append(a) for a in LL]
# print(custom_append(L, 6, 3))


# m = 3
# delta = -2
# graph = pa.pa_generator(50000, m, delta, 0)
# l = num_triangles_over_time(graph)
# X = np.log(range(len(l)//2, 50000))
# y = np.log(l[len(l)//2 :])
# model = np.polyfit(X,y,1)
# print(model)
# print(tri_exponent(m,delta))
# plt.plot(l)
# plt.show()
# plt.loglog(l)
# plt.grid(True)
# plt.show()


# =============================================================================
# betti_0 = []
# betti_1 = []
#
# for i in range(100):
#     g = pa.pa_generator(i, 4, -1, 4, 1)
#     mat = get_age_matrix(g)
#     dgms = ripser(mat, distance_matrix = True)['dgms']
#     betti_0.append(len(dgms[0]))
#     betti_1.append(len(dgms[1]))
#
# plt.plot(range(100), betti_0)
# plt.plot(range(100), betti_1)
# =============================================================================

# Chunyin's experiment
# This choice of seed gives mortal 1D cycles and has 2D holes
# =============================================================================
# g = pa.pa_generator(10, 4, -2, 1, 1)
# mat = get_age_matrix(g)
# print(mat)
# dgms = ripser(mat, distance_matrix = True, maxdim = 3)['dgms']
# plot_diagrams(dgms, show=True)
# print(dgms)
# =============================================================================
####


# This is a matrix that get_age_matrix should return
# mat = np.array(
#     [[1, 2, 0, 0],
#      [2, 2, 3, 4],
#      [0, 3, 3, 0],
#      [0, 4, 0, 4]]
#     )

# This should work as well.
# mat = np.array(
#     [[1, 2, 0, 0],
#       [2, 2, 3, 4],
#       [0, 0, 3, 0],
#       [0, 0, 0, 4]]
#     )

# This does not work, so Ripser only cares about upper diagonal entries.
# mat = np.array(
#     [[1, 0, 0, 0],
#       [2, 2, 0, 0],
#       [0, 3, 3, 0],
#       [0, 4, 0, 4]]
#     )

# This shows off-diagonal entries should be the time at which the edge appears.
# In particular, we should NOT divide off-diagonal entries by 2. (why?)
# mat = np.array([
#     [1, 4],
#     [0, 2]
#     ])


# =============================================================================
# dgms = ripser(mat, distance_matrix = True)['dgms']
# print(len(dgms[0]))
# print(len(dgms[1]))
# plot_diagrams(dgms, show=True)
# =============================================================================
