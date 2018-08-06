#!/usr/bin/env python

# ptr.py
# Created by Hayden Helm, Joshua Agterberg in June 2018. Adapted from Youngser Park.

import numpy as np
import networkx
from scipy.stats import rankdata

def pass_to_ranks(G, nedges = 0):
    """
    Passes an adjacency matrix to ranks.

    Inputs
        G - A networkx graph or 1 x n nd array
    Outputs
        PTR(G) - The passed to ranks version of the adjacency matrix of G
    """

    if type(G) == networkx.classes.graph.Graph:
        nedges = len(G.edges)
        edges = np.repeat(0, nedges)
        #loop over the edges and store in an array
        if networkx.is_weighted(G):
            j = 0
            for u, v, d in G.edges(data=True):
                edges[j] = d['weight']
                j += 1

            ranked_values = rankdata(edges)
            #loop through the edges and assign the new weight:
            j = 0
            for u, v, d in G.edges(data=True):
                #edges[j] = (ranked_values[j]*2)/(nedges + 1)
                d['weight'] = ranked_values[j]*2/(nedges + 1)
                j += 1
        
        return networkx.to_numpy_array(G)

    elif type(G) == np.ndarray:
        n, _ = G.shape
        unraveled_sim = G.ravel().copy()
        sorted_indices = np.argsort(unraveled_sim)

        if nedges == 0: # Defaulted to (n choose 2), matrix assumed to be symmetric
            E = int((n**2 - n)/2) # or E = int(len(single)/a1_sim.shape[0])
            for i in range(E):
                unraveled_sim[sorted_indices[(n - 2) + 2*(i + 1)]] = i/E
                unraveled_sim[sorted_indices[(n - 2) + 2*(i + 1) + 1]] = i/E

        else:
            for i in range(nedges):
                unraveled_sim[sorted_indices[-2*i - 1]] = (nedges - i)/nedges # assumes symmetric (undirected) matrix
                unraveled_sim[sorted_indices[-2*i - 2]] = (nedges - i)/nedges

            for i in range(n**2 - int(2*nedges)):
                unraveled_sim[sorted_indices[i]] = 0 # set rest of edges to 0

        ptred = unraveled_sim.reshape((n,n)) # back to similarity mat

        return ptred

def edge_count(A):
    n, _ = A.shape
    c = 0
    
    for i in range(n):
        for j in range(i, n):
            if A[i,j] != 0:
                c += 1
                
    return c