#!/usr/bin/env python

# omni.py

import numpy as np
import networkx
import pandas as pd

def omni_matrix(list_of_sim_matrices, off_diag = "mean"):
    """
    Inputs
        list_of_sim_matrices - The adjacencies to create the omni for
        off_diag = Metric used for off diagonals

    Returns
        omni - The omni matrix of the list
    """

    M = len(list_of_sim_matrices)
    n = len(list_of_sim_matrices[0])
    omni = np.zeros(shape = (M*n, M*n))

    for i in range(M):
        for j in range(i, M):
            for k in range(n):
                for m in range(k + 1, n):
                    if i == j:
                        omni[i*n + k, j*n + m] = list_of_sim_matrices[i][k, m] 
                        omni[j*n + m, i*n + k] = list_of_sim_matrices[i][k, m] # symmetric
                    else:
                        if off_diag == "mean":
                            omni[i*n + k, j*n + m] = (list_of_sim_matrices[i][k,m] + list_of_sim_matrices[j][k,m])/2
                            omni[j*n + m, i*n + k] = (list_of_sim_matrices[i][k,m] + list_of_sim_matrices[j][k,m])/2
    return omni

def get_attributes(G):
    if type(G) != networkx.classes.graph.Graph:
        raise TypeError("networkx.classes.graph.Graph only")

    if len(G) == 0:
        raise ValueError("Graph has no nodes")

    attributes = list(G.nodes[0])
    attr_values = []
    to_int = []
    for attr in attributes:
        temp = list(networkx.get_node_attributes(G, attr).values())
        if temp[0] == int(temp[0]) and temp[-1] == int(temp[-1]):
            to_int.append(attr)
        else:
            temp = np.array([float(i) for i in temp])
        attr_values.append(temp)

    attr_values = np.array(attr_values).T

    attr_df = pd.DataFrame(data = attr_values, columns = attributes)
    attr_df[to_int] = attr_df[to_int].astype(int)

    return attr_df


