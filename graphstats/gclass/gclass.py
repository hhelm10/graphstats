#!/usr/bin/env python

# gclass.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as MVN
import numpy as np
from networkx import Graph

def gaussian_classification(X, seeds, labels):
    """
    Gaussian classification (i.e. seeded gaussian "clustering").

    Inputs
        X - An n x d feature numpy array
        seeds - The training nodes
        labels - The classes for the training nodes
    Returns
        labels - Learned (MAP) class labels for each vertex
    """

    embedding = X.copy()

    seeds = np.array([int(i) for i in seeds])
    labels = np.array([int(i) for i in labels])

    unique_labels, label_counts = np.unique(labels, return_counts = True)
    K = len(unique_labels)

    n, d = embedding.shape

    if int(K) < d:
        embedding = embedding[:, :K].copy()
        d = int(K)

    ENOUGH_SEEDS = True # For full estimation

    #get unique labels
    unique_labels, label_counts = np.unique(labels, return_counts = True)
    for i in range(K):
        if label_counts[i] < d*(d + 1)/2:
            ENOUGH_SEEDS = False
            break

    pis = label_counts/len(seeds)

    #reindex labels if necessary
    for i in range(len(labels)): # reset labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        labels[i] = int(itemindex)

    #gather the meanss
    x_sums = np.zeros(shape = (K, d))

    for i in range(len(seeds)):
        temp_feature_vector = embedding[i, :]
        temp_label = labels[i]
        x_sums[temp_label, :] += temp_feature_vector

    estimated_means = [x_sums[i,:]/label_counts[i] for i in range(K)]

    mean_centered_sums = np.zeros(shape = (K, d, d))

    for i in range(len(seeds)):
        temp_feature_vector = embedding[i, :].copy()
        temp_label = labels[i]
        mean_centered_feature_vector = temp_feature_vector - estimated_means[labels[i]]
        temp_feature_vector = np.reshape(temp_feature_vector, (len(temp_feature_vector), 1))
        mcfv_squared = temp_feature_vector.dot(temp_feature_vector.T)
        mean_centered_sums[temp_label, :, :] += mcfv_squared
    
    if ENOUGH_SEEDS:
        estimated_cov = np.zeros(shape = (K, d, d))
        for i in range(K):
            estimated_cov[i] = mean_centered_sums[i,:]/(label_counts[i] - 1)
    else:
        estimated_cov = np.zeros(shape = (d,d))
        for i in range(K):
            estimated_cov += mean_centered_sums[i, :]*(label_counts[i] - 1)
        estimated_cov = estimated_cov / (n - d)

    PD = True
    eps = 0.001
    if ENOUGH_SEEDS:
        for i in range(K):
            try:
                eig_values = np.linalg.svd(estimated_cov[i, :, :])[1]
                if len(eig_values) > len(eig_values[eig_values > -eps]):
                    PD = False
                    break
            except:
                PD = False
                break

    means = container.ndarray(estimated_means)
    covariances = container.ndarray(estimated_cov)

    if PD and ENOUGH_SEEDS:
        for i in range(len(testing_nodeIDs)):
        #for i in range(len(nodeIDs)):
            #temp = np.where(nodeIDs == int(testing_nodeIDs[i]))[0][0]
            #temp = i
            weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], means[j], covariances[j, :, :]) for j in range(K)])
            label = np.argmax(weighted_pdfs)
            final_labels[i] = int(label)
    else:
        for i in range(len(testing_nodeIDs)):
            #temp = np.where(nodeIDs == int(testing_nodeIDs[i]))[0][0]
            weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], means[j], covariances) for j in range(K)])
            label = np.argmax(weighted_pdfs)
            final_labels[i] = int(label)

    return final_labels