#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

import numpy as np
import networkx
from sklearn.decomposition import TruncatedSVD

def adj_spectral_embedding(A, max_dim = 2, eig_scale = 0.5, return_spectrum = True, acorn = 1234):
    """
    Inputs
        A - A numpy array or networkx graph
    Outputs
        eig_vectors - The scaled (or unscaled) eigenvectors
    """
    np.random.seed(acorn)
    n, d = A.shape

    if n < d:
        A = A[:,:n].copy()
    elif d < n:
        A = A[:d, :].copy()

    if type(A) == networkx.classes.graph.Graph:
        A = networkx.to_numpy_array(A)

    tsvd = TruncatedSVD(n_components = min(max_dim, n - 1))
    tsvd.fit(A)

    eig_vectors = tsvd.components_.T
    eig_values = tsvd.singular_values_

    X_hat = eig_vectors[:, :d].copy()

    X_hat = eig_vectors.dot(np.diag(eig_values**eig_scale))

    if return_spectrum:
        return X_hat, eig_values
    else:
        return X_hat