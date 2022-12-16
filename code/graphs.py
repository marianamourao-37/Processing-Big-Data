# knn + neighbours
# draw_adjacency_graph

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np 

def knnregressor(features, target, n_neighbors):
    
    knn = KNeighborsRegressor(n_neighbors, algorithm= 'brute')
    
    knn.fit(features.T, target.T)

    grafo = knn.kneighbors_graph(features.T, n_neighbors, mode = 'distance')

    g_sparse_matrix = 0.5 * (grafo + grafo.T)

    return knn, g_sparse_matrix

def draw_adjacency_graph(sparse_matrix,
    clustering = 'False',                     
    node_color=[], 
    layout='graphviz',
    prog_parameter = 'neato', 
    node_size=60):

    graph = nx.from_scipy_sparse_matrix(sparse_matrix)
    
    if clustering:
        sparse_array = sparse_matrix.toarray()
        sparse_array[sparse_array == 0] = np.nan

        d_mean = np.nanmean(sparse_array)
        d_std = np.nanstd(sparse_array)
        threshold = d_mean + 2*d_std
        big_edge = np.argwhere(sparse_array > threshold)

        #print(big_edge)

        for edge in big_edge:
            if edge in graph.edges():
                graph.remove_edge(edge[0], edge[1])
    
    plt.figure(figsize=(15, 12))
    plt.grid(False)
    plt.axis('off')

    if layout == 'graphviz':
        pos = graphviz_layout(graph, prog = prog_parameter)
    else:
        pos = nx.spring_layout(graph)

    if not node_color:
        node_color='blue'
        
    nx.draw_networkx_nodes(graph, pos,
                           node_color = node_color, 
                           alpha = 0.6, 
                           node_size = node_size, 
                           cmap = plt.get_cmap('autumn'))
    
    nx.draw_networkx_edges(graph, pos, alpha = 0.5)
    plt.show()

