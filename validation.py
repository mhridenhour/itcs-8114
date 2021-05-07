import networkx as nx
import random
import numpy as np
import time
from statistics import mean

import graph_algos
import network

def main():    
    G = nx.erdos_renyi_graph(1000, 0.3)
    print('============== closness centrality ==========')
    r = network.closeness_centrality(G)
    nx_r = nx.algorithms.closeness_centrality(G)
    for node in [0,1,2,3,4,5,6,7,8,9]:
        print('\ncomputing closeness centrality for node {}...'.format(node))
        print('our result: {}'.format(r[node]))
        print('networkx result: {}'.format(nx_r[node]))
    print()

    print('============== clustering coefficient ==========')
    r = network.network_clustering(G)
    nx_r = nx.algorithms.clustering(G)
    for node in [0,1,2,3,4,5,6,7,8,9]:
        print('\ncomputing clustering coefficeint for node {}...'.format(node))
        print('our result: {}'.format(r[node]))
        print('networkx result: {}'.format(nx_r[node]))
    print()

    print('============== avg clustering coefficient ==========\n')
    print('our result: {}'.format(network.average_clustering(G)))
    print('networkx result: {}'.format(nx.algorithms.average_clustering(G)))
    print()

    print('============== diameter ==========\n')
    print('our result: {}'.format(network.network_diameter(G)))
    print('networkx result: {}'.format(nx.algorithms.diameter(G)))
    print()

    print('============== k-core ==========\n')
    _, core = graph_algos.matula_beck_degen_order(G)
    print('our result: {}'.format(core))
    nx_core_dict = nx.core_number(G)
    nx_core = 0
    for _, v in nx_core_dict.items():
        if v > nx_core:
            nx_core = v
    print('networkx result: {}'.format(nx_core))
    print()

    print('============== avg shortest path wrt node ==========')
    for node in [0,1,2,3,4,5,6,7,8,9]:
        print('\ncomputing avg shortest path for node {}...'.format(node))
        my_dist = graph_algos.avg_shortest_path_node(G, node)
        print('our result: {}'.format(my_dist))

        nx_dict = nx.shortest_path(G, node)
        nx_lengths = []
        for _, v in nx_dict.items():
            nx_lengths.append(len(v)-1)
        print('networkx result: {}'.format(mean(nx_lengths)))
    print()

    print('============== pagerank ==========')   
    DiG = G.to_directed()
    r = graph_algos.my_pagerank(DiG, 15)
    nx_r = nx.pagerank(DiG) 
    for node in [0,1,2,3,4,5,6,7,8,9]:
        print('\ncomputing pagerank for node {}...'.format(node))
        print('our result: {}'.format(r[node]))
        print('networkx result: {}'.format(nx_r[node]))

if __name__ == '__main__':
    main()