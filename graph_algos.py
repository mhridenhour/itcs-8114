import networkx as nx
import time
import numpy as np

## K CORE STUFF
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

def asc_order_nodes(G):
    H = G.copy()
    j = len(G.nodes())
    output_list = []

    while j >= 1:
        v_j, d_v = get_min_degree_node(H)
        output_list.append(v_j)
        H.remove_node(v_j)
    
    return output_list

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


def shortest_path(G, src_node):
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

    return distances




    

    

def main():
    G = nx.karate_club_graph()
    '''
    _, core = get_k_core(G, 4)
    print(core)
    print()
    kg = nx.k_core(G, k=4)
    print(kg.nodes)
    '''
    dist = shortest_path(G, 0)
    print(dist)
    print(nx.shortest_path(G, source=0,target=26))

if __name__ == '__main__':
    main()