import networkx as nx
import random
import numpy as np

def closeness_centrality(G):
    """Returns the closness centality of each node of the graph G.

    In a connected graph,closeness centrality (or closeness) of a node 
    is a measure of centrality in a network, calculated as the sum of the 
    length of the shortest paths between the node and all other nodes in the graph. 
    Thus the more central a node is, the closer it is to all other nodes.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.
    """


    path_length = nx.single_source_shortest_path_length
    nodes = G.nodes()
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G,n)
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp)-1.0) / totsp

            # normalize to number of nodes-1 in connected part
            s = (len(sp)-1.0) / ( len(G) - 1 )
            closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    return closeness_centrality

def network_clustering(G):
    """Returns the clustering coefficient of each node of the graph G.

    The diameter is the maximum eccentricity.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with clustering coefficient as the value.
    """

    cc = {}
    for node in G.nodes():
        neighbours=[n for n in nx.neighbors(G,node)]
        n_neighbors=len(neighbours)
        n_links=0
        if n_neighbors>1:
            for node1 in neighbours:
                for node2 in neighbours:
                    if G.has_edge(node1,node2):
                        n_links+=1
            n_links/=2 #because n_links is calculated twice
            clustering_coefficient=n_links/(0.5*n_neighbors*(n_neighbors-1))
            cc[node]=clustering_coefficient
    return cc

def average_clustering(G, trials=10000):
    n = len(G)
    triangles = 0
    random_trials = [int(random.random() * n) for i in range(trials)]
    for i in random_trials:
        nbrs = [n for n in G.neighbors(i)]
        if len(nbrs) < 2:
            continue
        u, v = random.sample(nbrs, 2)
        if u in G[v]:
            triangles += 1
    return triangles / float(trials)


# floyd_warshall implementation
def floyd_warshall(M, n):
    distance = list(map(lambda i: list(map(lambda j: j, i)), M))

    # Adding vertices individually
    for k in range(n):
        for i in range(n):
            for j in range(n):
                minval = min(distance[i][j], distance[i][k] + distance[k][j])
                distance[i][j] = minval
    return distance

def network_diameter(G):
    """Returns the diameter of the graph G.

    The diameter is the maximum eccentricity.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    Returns
    -------
    d : integer
       Diameter of graph
    """
    A = nx.to_numpy_array(G, multigraph_weight=min, weight='weight', nonedge=np.inf )
    n, m = A.shape
    np.fill_diagonal(A, 0)
    dist = floyd_warshall(A,n)
    dia = dist[0][0]
    for i in range(n):
        for j in range(n):
            if(dist[i][j] >= dia):
                dia  = dist[i][j]
            
    return dia
    
    
def main():    
    G = nx.karate_club_graph()
    print('============== Closness Centrality ==========')
    print(closeness_centrality(G))
    print()
    print(nx.algorithms.closeness_centrality(G))
    print()
    print('============== clustering coefficient ==========')
    print(nx.algorithms.clustering(G))
    print()
    print(network_clustering(G))
    print()
    print('============== avg clustering coefficient ==========')
    print(nx.algorithms.average_clustering(G))
    print()
    print(average_clustering(G))
    print()
    print('============== Diameter ==========')
    print(network_diameter(G))
    print()
    print(nx.algorithms.diameter(G))

if __name__ == '__main__':
    main()