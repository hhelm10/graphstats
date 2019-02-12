#!/usr/bin/env python

# lse.py
# Copyright (c) 2017. All rights reserved.

import numpy as np
import networkx
from sklearn.decomposition import TruncatedSVD

def lap_spectral_embedding(A, type_ = "DAD", d_scale = 0.5, max_dim = 2, eig_scale = 0.5, return_spectrum = True, acorn = 1234):
    """
    Inputs
        A - A numpy array or networkx graph
    Outputs
        eig_vectors - The scaled (or unscaled) eigenvectors
    """
    
    np.random.seed(acorn)

    if type(A) == networkx.classes.graph.Graph:
        A = networkx.to_numpy_array(A)

    if type_ == "DAD":
        degrees = np.array([np.sum(A[i, :]) for i in range(A.shape[0])])
        degrees_diag = np.diag(degrees)

        D = np.linalg.pinv(degrees_diag**d_scale)
        L = D @ A @ D
    elif type_ == "D-A":
        L = np.diag(A.sum(axis=1)) - A

    tsvd = TruncatedSVD(n_components = max_dim)
    tsvd.fit(L)

    eig_vectors = tsvd.components_.T
    eig_values = tsvd.singular_values_

    eig_vectors = eig_vectors.dot(np.diag(eig_values)**eig_scale)

    if return_spectrum:
        return eig_vectors, eig_values
    else:
        return eig_vectors