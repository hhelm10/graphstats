#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

import numpy as np
import networkx
from sklearn.decomposition import TruncatedSVD

def adj_spectral_embedding(A, max_dim = int(np.floor(A.shape[0]/10)), eig_scale = 0.5, return_spectrum = True, acorn = 1234)
    """
    Inputs
        A - A numpy array or networkx graph
    Outputs
        eig_vectors - The scaled (or unscaled) eigenvectors
    """
    np.random.seed(acorn)

    if type(A) == networkx.classes.graph.Graph:
        A = networkx.to_numpy_array(A)

    tsvd = TruncatedSVD(n_components = max_dim)
    tsvd.fit(A)

    eig_vectors = tsvd.components_.T
    eig_values = tsvd.singular_values_

    if scaled:
        eig_vectors = eig_vectors.dot(diag(eig_values)**eig_scale)

    if return_spectrum:
        return eig_vectors, eig_values
    else:
        return eig_vectors