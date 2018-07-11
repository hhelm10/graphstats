#!/usr/bin/env python

# gclust.py
# Copyright (c) 2017. All rights reserved.

import numpy as np
from sklearn.mixture import GaussianMixture

def gaussian_clustering(X, max_clusters = 2, min_clusters = 1, acorn = 1234):
    """
    Inputs
        X - n x d feature matrix; it is assumed that the d features are ordered
        max_clusters - The maximum number of clusters
        min_clusters - The minumum number of clusters

    Outputs
        Predicted class labels that maximize BIC
    """
    np.random.seed(acorn)

    if type(X) != np.ndarray:
        raise TypeError("numpy.ndarray only")

    if X.ndim < 2:
        raise TypeError("n x d, d > 1 numpy.ndarray only")

    n, d = X.shape

    max_clusters = int(round(max_clusters))
    min_clusters = int(round(min_clusters))

    if max_clusters < d:
        X = X[:, :max_clusters].copy()

    cov_types = ['full', 'tied', 'diag', 'spherical']

    clf = GaussianMixture(n_components = min_clusters, covariance_type = 'spherical')
    clf.fit(X)
    BIC_max = -clf.bic(X)
    cluster_likelihood_max = min_clusters
    cov_type_likelihood_max = "spherical"

    for i in range(min_clusters, max_clusters + 1):
        for k in cov_types:
            clf = GaussianMixture(n_components=i, 
                                covariance_type=k)

            clf.fit(X)

            current_bic = -clf.bic(X)
            #print(i, k, current_bic)

            if current_bic > BIC_max:
                BIC_max = current_bic
                cluster_likelihood_max = i
                cov_type_likelihood_max = k

    #print(cluster_likelihood_max, cov_type_likelihood_max)
    clf = GaussianMixture(n_components = cluster_likelihood_max,
                    covariance_type = cov_type_likelihood_max)
    clf.fit(X)

    predictions = clf.predict(X)
    predictions = np.array([int(i) for i in predictions])

    return predictions