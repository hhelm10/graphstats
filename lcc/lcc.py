from networkx import Graph
import networkx
import numpy as np

def largest_connected_component(G):
    """
    Input
        G: an n x n matrix or a networkx Graph 
    Output
        The largest connected component of g

    """

    if type(G) == np.ndarray:
        if G.ndim == 2:
            if G.shape[0] == G.shape[1]: # n x n matrix
                G = Graph(G)
            else:
                raise TypeError("Networkx graphs or n x n numpy arrays only") 

    subgraphs = [G.subgraph(i).copy() for i in networkx.connected_components(G)]

    G_connected = []
    for i in subgraphs:
        if len(i) > len(G_connected):
            G_connected = i

    return G_connected