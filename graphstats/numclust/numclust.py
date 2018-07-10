#!/usr/bin/env python

# numclust.py
# Copyright (c) 2017. All rights reserved.

import numpy as np
from sklearn.mixture import GaussianMixture

def num_clust(X, max_clusters = round(np.log(X.shape[0]))):
    """
    Inputs
        X - n x d feature matrix
    Return
        An array with the max BIC and AIC values for each number of clusters (1, .., max_clusters)
    """
    max_clusters = int(max_clusters)

    cov_types = ['full', 'tied', 'diag', 'spherical']

    example = ('number of clusters', 'BIC', 'AIC')

    BICs = []

    AICs = []

    results = [example]

    for i in range(1, max_clusters + 1):

        clf = GaussianMixture(n_components=i, 
                                covariance_type='spherical')
        clf.fit(X)
        temp_max_BIC, temp_max_AIC = -clf.bic(X), -clf.aic(X)
        for k in cov_types:
            clf = GaussianMixture(n_components=i, 
                                covariance_type=k)

            clf.fit(X)

            temp_BIC, temp_AIC = -clf.bic(X), -clf.aic(X)

            if temp_BIC > temp_max_BIC:
                temp_max_BIC = temp_BIC

            if temp_AIC > temp_max_AIC:
                temp_max_AIC = temp_AIC

        results.append((i, temp_max_BIC, temp_max_AIC))
        BICs.append(temp_max_BIC)
        AICs.append(temp_max_AIC)

    BICs2 = list(BICs)
    AICs2 = list(AICs)

    KBICs = []

    KAICs = []

    while len(BICs2) > 0:
        temp, temp_index = max(BICs2), np.argmax(BICs2)
        index = BICs.index(temp)
        KBICs.append(index + 1)
        BICs2.pop(temp_index)

        temp, temp_index = max(AICs2), np.argmax(AICs2)
        index = AICs.index(temp)
        KAICs.append(index + 1)
        AICs2.pop(temp_index)

    KBICs = ['Ranked number of clusters (BIC)'] + KBICs
    KAICs = ['Ranked number of clusters (AIC)'] + KAICs

    outputs = [KBICs, KAICs, results]

    return outputs