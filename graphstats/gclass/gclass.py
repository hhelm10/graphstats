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

def gaussian_classification(X, seeds, labels, update_priors = False):
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
    n, d = embedding.shape

    seeds = np.array([int(i) for i in seeds])
    labels = np.array([int(i) for i in labels])

    unique_labels, label_counts = np.unique(labels, return_counts = True)
    K = len(unique_labels)

    if int(K) < d:
        embedding = embedding[:, :K].copy()
        d = int(K)

    ENOUGH_SEEDS = True # For full estimation
    for i in range(K):
        if label_counts[i] < d*(d + 1)/2:
            ENOUGH_SEEDS = False
            break

    pis = label_counts/len(seeds)

    for i in range(len(labels)): # reindex labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        labels[i] = int(itemindex)

    #gather the meanss
    x_sums = np.zeros(shape = (K, d))

    for i in range(len(seeds)):
        temp_feature_vector = embedding[seeds[i], :]
        temp_label = labels[i]
        x_sums[temp_label, :] += temp_feature_vector

    estimated_means = np.array([x_sums[i,:]/label_counts[i] for i in range(K)])

    mean_centered_sums = np.zeros(shape = (K, d, d))

    for i in range(len(seeds)):
        temp_feature_vector = embedding[seeds[i], :].copy()
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

    final_labels = np.zeros(n)

    if PD and ENOUGH_SEEDS:
        for i in range(n):
            if not update_priors:
        #for i in range(len(nodeIDs)):
            #temp = np.where(nodeIDs == int(testing_nodeIDs[i]))[0][0]
            #temp = i
                weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)])
                label = np.argmax(weighted_pdfs)
                final_labels[i] = int(label)
            else:
                return


    else:
        for i in range(n):
            if not update_priors:
            #temp = np.where(nodeIDs == int(testing_nodeIDs[i]))[0][0]
                weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)])
                label = np.argmax(weighted_pdfs)
                final_labels[i] = int(label)
            else:
                return
    #print(pis, estimated_means, estimated_cov)

    return final_labels

def permutation_error(perm1, value1, perm2, value2, include = "all"):
    perm1 = np.array(perm1)
    perm2 = np.array(perm2)
    
    set1 = set(perm1)
    set2 = set(perm2)
    if len(set1) != len(set2):
        raise ValueError("Permutations of different objects")
    elif set1 != set2:
        list1 = np.zeros(len(perm1))
        list2 = np.zeros(len(perm2))
        temp1 = np.argsort(perm1)
        temp2 = np.argsort(perm2)
        print(temp1, temp2)
        for i in range(len(perm1)):
            list1[temp1[i]] = i + 1 
            list2[temp2[i]] = i + 1

    sim_c = 0    
    for i in range(len(list1)):
        if include == "errors":
            if list1[i] != list2[i]:
                sim_c += np.sqrt((value1[i] - value2[i])**2)/value1[i]
        elif include == "all":
            sim_c += np.sqrt((value1[i] - value2[i])**2)/value1[i]
            
    sim = 1 - sim_c/len(perm1)
     
    return sim

def calculate_priors(pis, values):
    if len(values) == 0:
        return pis
    elif len(values) == 1:
        return pis

    pis_ = np.array(pis)
    values_ = np.array(values)

    new_pi = np.array([pis[i]*values[i] for i in range(len(pis))]) / pis.dot(values)
    return new_pi

def estimate_means(A, seeds, labels, unlabeled = [], return_argsort = False):
    new_labels = np.zeros(labels)

    unique_labels, label_counts = np.unique(labels, return_counts = True)
    K = len(unique_labels)

    means = np.zeros(shape = (K, K))

    for i in range(len(labels)): # reindex labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        new_labels[i] = int(itemindex)

    if len(unlabeled) == 0:
        for k in range(len(seeds)):
            for l in range(k + 1, len(seeds)):
                means[int(new_labels[k]), int(new_labels[l])] += A[seeds[k], seeds[l]] # update appropriate means
                        
        for i in range(K):
            for j in range(i, K):
                means[i,j] / (label_counts[i] * label_counts[j])
                means[j, i] = means[i,j]
    elif unlabeled[0] < 0:
        raise ValueError("Negative index")
    else:
        for k in range(len(seeds)):
            means[int(new_labels[k]), int(new_labels[l])] += A[seeds[k], seeds[l]]

        zero_index = np.where(means == 0)[0]

        if len(zero_index) == 0:
            raise ValueError("Insufficient seeds")
        else:
            for i in range(K):
                for j in range(i, K):
                    means[i,j] / (label_counts[i] * label_counts[j])
                    means[j, i] = means[i,j]
        for i in range(K):
            for j in range(len(seeds)):
                means[i, int(new_labels[j])] += A[unlabeled[0], seeds[j]]

            means[i,k] / len(label_counts[i] + 1)

    argsorts = means.copy()
    for i in range(K):
        argsorts[i,:] = np.argsort(means[i, :])

    if return_argsort:
        return means, argsorts
    else:
        return means

def choose(n, k):
    from math import factorial
    if n < k:
        return 0
    
    else:
        num = factorial(n)
        den = factorial(k)*factorial(n - k)
    return int(num/den)

