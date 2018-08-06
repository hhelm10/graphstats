#!/usr/bin/env python

# gclass.py
# Copyright (c) 2017. All rights reserved.
import os

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as MVN
from scipy.stats import norm
import numpy as np
from networkx import Graph

def gaussian_classification(X, seeds, labels, cov_type = "", sims = [], errors = False):
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

    if embedding.ndim == 1:
        n = len(embedding)
        d = 1
    else:
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

    #gather the means

    if d == 1:
        seeds_split = [[] for i in range(K)]
        for i in range(len(seeds)):
            temp_label = labels[i]
            seeds_split[temp_label].append(seeds[i])
        estimated_means = np.array([np.mean(k) for k in seeds_split])
        estimated_cov = np.array(np.var(k, ddof = 1) for k in seeds_split)
        if cov_type == "tied":
            temp = 0
            for k in range(K):
                temp += (len(seeds_split[k]) - 1)*estimated_cov[k]
            for k in range(K):
                temp/(n - K)
            estimated_cov = np.array([temp for k in K])

        final_labels = np.zeros(n)
        unlabeled = [i for i in range(n) if i not in seeds]

        if len(sims) == 0:
            for i in range(len(unlabeled)):
                weighted_pdfs = np.array([pis[j]*norm.pdf(embedding[i], estimated_means[j], estimated_cov[j]) for j in range(K)])
                label = np.argmax(weighted_pdfs)
                final_labels[i] = int(label)
        else:
            apriori = np.array([update_priors(np.array(pis), sims[i], errors = errors) for i in range(len(unlabeled))])
            for i in range(len(unlabeled)):
                weighted_pdfs = np.array([apriori[i][j]*norm.pdf(embedding[i], estimated_means[j], estimated_cov[j]) for j in range(K)])
                label = np.argmax(weighted_pdfs)
                final_labels[i] = int(label)

    else:
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

        unlabeled = [i for i in range(n) if i not in seeds]

        print(estimated_means, estimated_cov)

        if PD and ENOUGH_SEEDS:
            if len(sims) == 0:
                for i in range(len(unlabeled)):
                    weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)
            else:
                apriori = np.array([update_priors(np.array(pis), sims[i], errors = errors) for i in range(len(unlabeled))])
                print(apriori)
                for i in range(len(unlabeled)):
                    #print()
                    weighted_pdfs = np.array([apriori[i, j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)
        else:
            if len(sims) == 0:
                for i in range(len(unlabeled)):
                    weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)
            else:
                apriori = np.array([update_priors(np.array(pis), sims[i]) for i in range(len(unlabeled))])
                print(apriori)
                for i in range(len(unlabeled)):
                    weighted_pdfs = np.array([apriori[i,j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)])
                    #print(weighted_pdfs)
                    #print(np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)]))
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)

    return final_labels

def permutation_error(perm1, value1, perm2, value2, include = "all"):
    perm1 = np.array(perm1)
    perm2 = np.array(perm2)
    
    K = max(len(perm1), len(perm2))
    
    if K == 0 or K == 1:
        return
    
    zero = np.zeros(K)
    
    if sum(perm1 == zero) == K or sum(perm2 == zero) == K:
        return 0
    
    set1 = set(perm1)
    set2 = set(perm2)
    if len(set1) != len(set2):
        raise ValueError("Permutations of different objects")
    elif set1 != set2:
        list1 = np.zeros(len(perm1))
        list2 = np.zeros(len(perm2))
        temp1 = np.argsort(perm1)
        temp2 = np.argsort(perm2)
        for i in range(len(perm1)):
            list1[temp1[i]] = i + 1 
            list2[temp2[i]] = i + 1
    else:
        list1 = perm1.copy()
        list2 = perm2.copy()
        
    K = len(list1)
    #print(K)
    
    if K == 2:
        sim_c = 1
    else:
        sim_c = 0
    
    for i in range(K):
        if include == "errors":
            if list1[i] != list2[i]:
                if value1[i] > 0:
                    if K == 2:
                        sim_c += 1
                    else:
                        sim_c += abs(value1[i] - value2[i])/value1[i]
                else:
                    return -1
        elif include == "all":
            if value1[i] > 0:   
                sim_c += abs(value1[i] - value2[i])/value1[i]
            else:
                return -1
            
    #print(sim_c)
            
    return sim_c

def update_priors(pis, values, errors = False):
    if len(values) == 0:
        return pis
    elif len(values) == 1:
        return pis
    elif np.count_nonzero(values) < len(pis) - 1:
        return pis

    pis_ = np.array(pis)
    values_ = np.array(values)

    if errors:
        values_ = np.ones(len(pis)) - (values_/np.sum(values_))
        
    new_pi = np.array([pis[i]*values_[i] for i in range(len(pis))]) / pis.dot(values_)

    return new_pi

def estimate_means(A, seeds, labels, unlabeled = [], return_argsort = False):
    new_labels = np.zeros(len(labels))

    unique_labels, label_counts = np.unique(labels, return_counts = True)
    #print(label_counts)
    K = len(unique_labels)

    for i in range(len(labels)): # reindex labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        new_labels[i] = int(itemindex)

    if len(unlabeled) == 0:
        means = np.zeros(shape = (K, K))
        nedges = np.zeros(shape = (K, K))
        for k in range(len(seeds)):
            for l in range(k + 1, len(seeds)):
                temp = A[seeds[k], seeds[l]]
                if temp != 0:
                    means[int(new_labels[k]), int(new_labels[l])] += A[seeds[k], seeds[l]] # update appropriate means
                    nedges[int(new_labels[k]), int(new_labels[l])] += 1
                
        for i in range(K):
            zeros = K - np.count_nonzero(nedges[i, :])

            if zeros > 0:
                means[i, :] = np.zeros(K)
            else:
                for j in range(i, K):
                    nedges[j, i] = nedges[i,j]
                    means[i,j] = means[i,j] / nedges[i,j]
                    means[j, i] = means[i,j]
                    nedges[j, i] = nedges[i,j]
                
        argsorts = means.copy()
        
        for i in range(K):
            if K - np.count_nonzero(means[i, :]) > 0:
                argsorts[i, :] = np.zeros(K)
            else:
                argsorts[i, :] = np.argsort(means[i, :])
        
    elif unlabeled[0] < 0:
        raise ValueError("Negative index")
    else:
        n = len(unlabeled)
        
        means = np.zeros(shape = (n, K))
        argsorts = means.copy()
        nedges = np.zeros(shape = (n, K))
        
        for i in range(n):
            for k in range(len(seeds)):
                temp = A[seeds[k], unlabeled[i]]
                if temp != 0:
                    means[i, int(new_labels[k])] += A[seeds[k], unlabeled[i]]
                    nedges[i, int(new_labels[k])] += 1

            zero_index = np.where(nedges[i, :] == 0)[0]

            if len(zero_index) > 0:
                means[i, :] = np.zeros(K)
            else:
                for j in range(K):
                    #print(means[i,j], nedges[i,j])
                    means[i,j] = means[i, j] / nedges[i, j]
                argsorts[i, :] = np.argsort(means[i, :])

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

def strip_weights(A, return_weights = False):
    n, _ = A.shape
    B = np.zeros(shape = (n,n))
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i,j] != 0:
                weights.append(A[i,j])
                B[i,j] = 1
                B[j,i] = 1 # assumed symmetry
    weights = np.array(weights)
    if return_weights:
        return B, weights
    else:
        return B  

def permutation_similarity(perm1, value1, perm2, value2, include = "all"):
    perm1 = np.array(perm1)
    perm2 = np.array(perm2)
    
    K = max(len(perm1), len(perm2))
    
    zero = np.zeros(K)
    
    if sum(perm1 == zero) == K or sum(perm2 == zero) == K: # check to see if the perm is all 0s (default value for insufficient seeds)
        return 0
    
    set1 = set(perm1) 
    set2 = set(perm2)
    if len(set1) != len(set2):
        raise ValueError("Permutations of different objects")
    elif set1 != set2:
        list1 = np.zeros(len(perm1)) # reset the permutations to 0 .. K - 1
        list2 = np.zeros(len(perm2))
        temp1 = np.argsort(perm1)
        temp2 = np.argsort(perm2)
        #for i in range(len(perm1)):
        #    list1[temp1[i]] = i + 1 # reset the permutations to 1 .. K
        #    list2[temp2[i]] = i + 1
    else:
        list1 = perm1.copy()
        list2 = perm2.copy()

    sim_c = 0
    for i in range(K):
        if include == "errors":
            if list1[i] != list2[i]:
                sim_c += abs(value1[i] - value2[i])/max(value1[i], value2[i])
        elif include == "all":
            sim_c += abs(value1[i] - value2[i])/max(value1[i], value2[i])
            
    sim = 1 - sim_c/K
    return sim