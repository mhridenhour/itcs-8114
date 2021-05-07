import networkx as nx
import time
import numpy as np
from statistics import mean

'''
Edgelist to nx Graph
'''
def edgelist_to_nx(filepath):
    G = nx.Graph()
    DiG = nx.DiGraph()

    with open(filepath) as f:
        lines = f.readlines()
        
        for line in lines:
            if line[0] == '#':
                # print(line)
                continue
            else:
                line = line.split('\t')
                G.add_edge(int(line[0].strip()), int(line[1].strip()))
                DiG.add_edge(int(line[0].strip()), int(line[1].strip()))

    return G, DiG

'''
K-Core: 
Our solution to the maximum k-core problem is via the Matula-Beck algortihm shown below.
We included an extra function that allows a user to input a k and generate a subgraph for that k.
'''
def matula_beck_degen_order(G, asc=True):
    H = G.copy()
    j = len(G.nodes())

    order = []
    max_k = 0

    while j >= 1:
        v_j, d_v = get_min_degree_node(H)
        max_k = max(d_v, max_k)
        H.remove_node(v_j)
        j -= 1
        if asc==True:
            order.append(v_j)
        else:
            order.insert(0, v_j)

    return order, max_k

def get_min_degree_node(G):
    min_node = None
    min_deg = 0
    for node in G.nodes():
        if min_node == None:
            min_node = node
        else:
            if G.degree[node] < G.degree[min_node]:
                min_node = node
                min_deg = G.degree[node]
            else:
                continue

    return min_node, min_deg

def get_k_core(G, k):
    _, max_k = matula_beck_degen_order(G)

    if k > max_k:
        print('unable to process, max k is {}'.format(max_k))
        return 0

    H = G.copy()
    min_deg = -np.inf

    while min_deg < k:
        min_node, min_deg = get_min_degree_node(H)
        if min_deg < k:
            H.remove_node(min_node)

    return(H, list(H.nodes()))


'''
Average shortest path 
'''
def avg_shortest_path_node(G, src_node):
    distances = {}
    distances[src_node] = 0
    unvisited = set(G.nodes()) - {src_node}
    for v in unvisited:
        distances[v] = np.inf

    queue = [src_node]
    
    while len(queue) > 0:
        
        u = queue.pop(0)
        nebs = set(G.neighbors(u))

        for v in (nebs & unvisited):
            distances[v] = distances[u] + 1
            queue.append(v)
            unvisited.remove(v)

    avg_dist = mean(distances[node] for node in distances.keys() if distances[node] != np.inf)
    
    return avg_dist

def avg_shortest_path_graph(G):
    out_list = []
    for node in G.nodes():
        out_list.append(avg_shortest_path_node(G, node))
    return mean(out_list)

'''
PageRank
'''

def my_pagerank(DiG, num_iters):
    ranks = dict()
    N = DiG.number_of_nodes()

    for node in DiG.nodes():
        ranks[node] = 1/N

    for _ in range(num_iters):
        temp_ranks = dict()
        for node in DiG.nodes():
            in_weight = 0
            for pred in DiG.predecessors(node):
                in_weight += ranks[pred]/len(list(DiG.successors(pred)))
            temp_ranks[node] = in_weight
        ranks = temp_ranks

    return ranks

def main():
    
    G = nx.erdos_renyi_graph(1000, 0.3)
    print('\nK-CORE')
    _, core = matula_beck_degen_order(G)
    print('computing k-core...')
    print()
    print('our result: {}'.format(core))
    nx_core_dict = nx.core_number(G)
    nx_core = 0
    for _, v in nx_core_dict.items():
        if v > nx_core:
            nx_core = v
    print('networkx result: {}'.format(nx_core))
    print('\n***************************************\n')
    # shortest path wrt node baseline
    print('SHORTEST PATH W.R.T. NODE')
    for node in [0,1,2,3,4,5,6,7,8,9]:
        print('\ncomputing distances for node {}...'.format(node))
        my_dist = avg_shortest_path_node(G, node)
        print('our result: {}'.format(my_dist))

        nx_dict = nx.shortest_path(G, node)
        nx_lengths = []
        for _, v in nx_dict.items():
            nx_lengths.append(len(v)-1)
        print('networkx result: {}'.format(mean(nx_lengths)))
    print('\n***************************************\n')
    print('PAGERANK')
    #pagerank
    DiG = G.to_directed()
    r = my_pagerank(DiG, 15)
    nx_r = nx.pagerank(DiG)
    for node in [0,1,2,3,4,5,6,7,8,9]:
        print('\ncomputing distances for node {}...'.format(node))
        print('our result: {}'.format(r[node]))
        print('networkx result: {}'.format(nx_r[node]))


if __name__ == '__main__':
    main()