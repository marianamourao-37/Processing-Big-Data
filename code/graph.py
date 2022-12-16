# knn + neighbours
# draw_adjacency_graph

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

def knnregressor(features, target, n_neighbors):
    
    knn = KNeighborsRegressor(n_neighbors, algorithm= 'brute')
    
    knn.fit(features, target)

    grafo = knn.kneighbors_graph(features, n_neighbors, mode = 'distance')

    g_sparse_matrix = 0.5 * (grafo + grafo.T)

    return knn, g_sparse_matrix

def draw_adjacency_graph(sparse_matrix,
    node_color=[], 
    layout='graphviz',
    prog_parameter = 'neato', 
    node_size=60):

    graph = nx.from_scipy_sparse_matrix(sparse_matrix)

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

