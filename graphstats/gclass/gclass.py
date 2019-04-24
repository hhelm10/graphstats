#!/usr/bin/env python

# gclass.py
# Copyright (c) 2017. All rights reserved.

from ..ase import adj_spectral_embedding
from ..dimselect import profile_likelihood
from ..sbm import adj_matrix
from ..ptr import pass_to_ranks

from itertools import zip_longest

import numpy as np
from networkx import Graph
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as MVN
from scipy.stats import norm
from numpy.random import uniform
from tqdm import tqdm

def gaussian_classification(X, seeds, labels, cov_type = "", sims = [], errors = False, return_likelihoods = False):
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

        estimated_means = np.array([np.mean(k) for k in embedding[np.array(seeds_split)]])
        estimated_cov = np.array([np.var(k, ddof = 1) for k in embedding[np.array(seeds_split)]])

        if cov_type == "tied":
            temp = 0
            for k in range(K):
                temp += (len(seeds_split[k]) - 1)*estimated_cov[k]
            for k in range(K):
                temp/(n - K)
            estimated_cov = np.array([temp for k in range(K)])

        final_labels = np.zeros(n)
        unlabeled = [i for i in range(n) if i not in seeds]

        if return_likelihoods:
            unlabeled_likelihoods = np.zeros(shape = (n, K))
            for i in range(n):
                weighted_pdfs = np.array([norm.pdf(embedding[i, 0], estimated_means[j], estimated_cov[j]**(1/2)) for j in range(K)])
                unlabeled_likelihoods[i, :] = weighted_pdfs
            return unlabeled_likelihoods
        else:
            if len(sims) == 0:
                for i in range(n):
                    weighted_pdfs = np.array([pis[j]*norm.pdf(embedding[i, 0], estimated_means[j], estimated_cov[j]**(1/2)) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)
            else:
                apriori = np.array([update_priors(np.array(pis), sims[i], errors = errors) for i in range(len(unlabeled))])
                for i in range(n):
                    weighted_pdfs = np.array([apriori[i][j]*norm.pdf(embedding[i, 0], estimated_means[j], estimated_cov[j]**(1/2)) for j in range(K)])
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

        #unlabeled = [i for i in range(n) if i not in seeds]

        if PD and ENOUGH_SEEDS:
            if len(sims) == 0:
                if return_likelihoods:
                    unlabeled_likelihoods = np.zeros(shape = (n, K))
                    for i in range(n):
                        weighted_pdfs = np.array([MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)])
                        unlabeled_likelihoods[i, :] = weighted_pdfs
                    return unlabeled_likelihoods
                else:
                    for i in range(n):
                        weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)])
                        label = np.argmax(weighted_pdfs)
                        final_labels[i] = int(label)
            else:
                apriori = np.array([update_priors(np.array(pis), sims[i], errors = errors) for i in range(n)])
                for i in range(n):
                    weighted_pdfs = np.array([apriori[i, j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)
        else:
            if len(sims) == 0:
                if return_likelihoods:
                    unlabeled_likelihoods = np.zeros(shape = (n, K))
                    for i in range(n):
                        weighted_pdfs = np.array([MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)])
                        unlabeled_likelihoods[i, :] = weighted_pdfs
                    return unlabeled_likelihoods
                else:
                    for i in range(n):
                        weighted_pdfs = np.array([pis[j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)])
                        label = np.argmax(weighted_pdfs)
                        final_labels[i] = int(label)
            else:
                apriori = np.array([update_priors(np.array(pis), sims[i]) for i in range(n)])
                for i in range(n):
                    weighted_pdfs = np.array([apriori[i,j]*MVN.pdf(embedding[i,:], estimated_means[j], estimated_cov) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = int(label)
            
    for i in range(len(seeds)):
        final_labels[seeds[i]] = labels[i]

    return final_labels

def get_weights(A, seeds, labels, index = -1):
    """
    A method to return a list of lists of edge weights.

    Inputs
        A - An n x n matrix
        seeds - The indices corresponding to the relevant weights
        labels - The class labels corresponding to the seeds
        index - The index of the object for which to grab weights (index = -1 means for all training data)
    Return
        weights - A list of lists of edge weights
    """

    unique_labels = np.unique(labels)
    K = len(unique_labels)

    new_labels = np.zeros(len(labels))

    for i in range(len(labels)): # reindex labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        new_labels[i] = int(itemindex)

    if index == -1:

        weights = [[[] for j in range(K - i)] for i in range(K)]

        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                min_ = int(min(new_labels[i], new_labels[j]))
                max_ = int(max(new_labels[i], new_labels[j]))
                diff = max_ - min_
                if A[seeds[i], seeds[j]] != 0:
                    weights[min_][diff].append(A[seeds[i], seeds[j]])

        symmetric = [[[] for j in range(K)] for i in range(K)]

        for i in range(K):
            for j in range(K - i):
                symmetric[i][i + j] = weights[i][j]
                symmetric[i + j][i] = symmetric[i][i + j]

        return symmetric
    else:
        weights = [[] for i in range(K)]

        for i in range(len(seeds)):
            if A[int(index), int(seeds[i])] != 0:
                weights[int(labels[i])].append(A[int(index), int(seeds[i])])

        return weights

def updated_priors_classification(X_hat, W, seeds, labels, method = "KS", logit_coefficient = 0, add_smoothing = 1, test = False, alpha = 0.1, params = "", model = "", acorn = 1234):
    """
    A method to perform updated priors classification.

    Inputs
        X_hat - An n x d array (typically a spectral embedding of a graph object)
        W - An n x n nd.array of weights
        seeds - An array of indices corresponding to training data
        labels - An array of labels corresponding to training data
        dist - True if W is a distance matrix
        pis - Class priors
        max_dim - The maximum embedding dimension if X_hat is undefined
        method - The method used to update priors
        add_smoothing - The amount to "smoooth" the updated priors
        test - True if testing for a difference in the means
        alpha - The Type 1 error (incorrectly rejection the null) threshold
        params
        model
        acorn - The random seed
    Return
        final_labels - Predicted class labels.
    """

    from math import log
    from scipy.stats import chi2

    order_ass = True

    unique_labels, label_counts = np.unique(labels, return_counts = True)
    pis = label_counts / len(labels)

    n = X_hat.shape[0]
    K = len(pis)
    unlabeled = [i for i in range(n) if i not in seeds]

    # print(unlabeled)

    if params == "" and method == "ranks":
        U = estimate_means(W, seeds, labels, unlabeled = unlabeled, return_argsort = True)
        params = estimate_means(W, seeds, labels, return_argsort = True)[1]

    if test:
        weights = get_weights(W, seeds, labels)
        order_ass, p_values = order_assumptions(weights, alpha, True)
        FS = -2*np.sum(np.log(p_values))
        #print(FS)
        p_meta = 1 - chi2.cdf(FS, 2*K)
        #print(p_meta)

    if method == "ranks":
        if test:
            if add_smoothing > 0: # If we want to transform our p-value (instead of just accept/reject)
                p_transformed = K*1000*p_meta
                #print(p_transformed, '\n')

                dissims = [np.array([footrule(params[k], U[1][i]) + K*p_transformed for k in range(K)]) for i in range(len(unlabeled))]
                sims = [np.array([1 - (dissims[i][k]/np.sum(dissims[i][:])) for k in range(K)]) for i in range(len(unlabeled))]

            else: # If we just accept or reject based on p-values
                if order_ass:
                    dissims = [np.array([footrule(params[k], U[1][i]) + 1 for k in range(K)]) for i in range(len(unlabeled))]
                    sims = [np.array([1 - (dissims[i][k]/np.sum(dissims[i][:])) for k in range(K)]) for i in range(len(unlabeled))]
                else:
                    sims = [np.array([1 for i in range(K)]) for i in range(len(unlabeled))]
        else:
            dissims = [np.array([footrule(params[k], U[1][i]) + add_smoothing for k in range(K)]) for i in range(len(unlabeled))]
            sims = [np.array([1 - (dissims[i][k]/np.sum(dissims[i][:])) for k in range(K)]) for i in range(len(unlabeled))]
        
    # elif method in ['AOL', 'LOA', 'joint']:
    #     sims = weight_likelihoods(W, seeds, labels, params = params, model = model, method = method)
    elif method == 'KS':
        seed_weights = get_weights(W, seeds, labels)
        weights = [get_weights(W, seeds, labels, int(unlabeled[i])) for i in range(len(unlabeled))]
        sims = [get_p_multiple_KS(weights[i], seed_weights, logit_coefficient = logit_coefficient) for i in range(len(unlabeled))]
    elif method == 'kNN':
        sims = [kNN_similarity(W[int(i), seeds], labels) for i in unlabeled]
    else:
        raise ValueError('unsupported method')

    new_pis = np.array([update_priors(pis, sims[i]) for i in range(len(unlabeled))])
    
    likelihoods = gaussian_classification(X_hat, seeds, labels, return_likelihoods = True)

    # print("X_hat:", X_hat[unlabeled[0]])
    # print("old priors:", new_pis[0])
    # print('similarities:', sims[0], np.argmax(sims[0]))
    # print('new_priors:', new_pis[0])
    # print('class conditionals:', likelihoods[unlabeled[0]])
    # print('old likelihoods:', pis * likelihoods[unlabeled[0]], np.argmax(pis * likelihoods[unlabeled[0]]))
    # print('new likelihoods:', likelihoods[unlabeled[0]] * new_pis[0], np.argmax(likelihoods[unlabeled[0]] * new_pis[0]))
    final_labels = np.zeros(n)
    for i in range(len(unlabeled)):
        final_labels[unlabeled[i]] = np.argmax(likelihoods[unlabeled[i]] * new_pis[i])

    for i in range(len(seeds)):
        final_labels[seeds[i]] = labels[i]

    return final_labels

def get_p_multiple_KS(collection_of_weights1, collection_of_weights2, logit_coefficient = 0):
    from scipy.stats import ks_2samp as KS
    from scipy.stats import chi2
    
    K = len(collection_of_weights1)

    for i in range(K):
        if len(collection_of_weights1[i]) == 0:
            return np.ones(K)

    all_ps = np.zeros(K)
    for i in range(K):
        ps = np.zeros(K)
        for j in range(K):
            ps[j] = KS(collection_of_weights1[j], collection_of_weights2[i][j])[1]
    
        FS = -2*np.sum(np.log(ps))
        
        all_ps[i] = 1 - chi2.cdf(FS, 2*K)
    

    return 1 / (1 + np.exp(-logit_coefficient * (all_ps - 0.5)))

def kNN_similarity(training_data_distances, labels, k = 5):
    # I assume that training_data_distances only contains ||X_{i}, X||, where b(X_{i}) is known
    sorted = np.argsort(training_data_distances)[:k] # sort the distances
    unique_labels = np.unique(labels)
    K = len(unique_labels) # find the number of labels

    nn_labels = labels[sorted] # take the labels corresponding to nearest neighbors
    proportions = np.array([np.sum(nn_labels == i)/k for i in unique_labels]) # normalize

    # print(proportions, np.argmax(proportions))

    return proportions

def kNN(W, seeds, labels, k = 5):
    seeds = np.array([int(i) for i in seeds])
    labels = np.array([int(i) for i in labels])

    n = W.shape[0]

    unique_labels, label_counts = np.unique(labels, return_counts = True)
    K = len(unique_labels)

    unlabeled = np.array([i for i in range(W.shape[0]) if i not in seeds])

    final_labels = -1*np.ones(n) # to be able to check that every node gets a label in {0, .., K-1}

    for i in unlabeled:
        final_labels[i] = np.argmax(kNN_similarity(W[i, seeds], labels, k))

    for i in range(len(seeds)):
        final_labels[seeds[i]] = labels[i] 

    return final_labels

def order_assumptions(weights, alpha, return_p):
    # Right now only works for K = 2
    from scipy.stats import mannwhitneyu as mwu
    T1, p1 = mwu(weights[0][0], weights[0][1])
    T2, p2 = mwu(weights[0][1], weights[1][1])
    #K = len(weights)
    #T = np.zeros(shape = (K, K))
    #p = np.zeros(shape = (K, K))
    
    if p1 < alpha and p2 < alpha:
        if return_p:
            return True, np.array([p1, p2])
        else:
            return True
    if return_p:
        return False, np.array([p1, p2])
    else:
        return False

def rank_classification(A, seeds, labels, order = ""):
    n = A.shape[0]
    unlabeled = [i for i in range(n) if i not in seeds]

    unique_labels = np.unique(labels)
    K = len(unique_labels)

    U = estimate_means(A, seeds, labels, unlabeled = unlabeled, return_argsort = True)

    if order == "":
        order = estimate_means(A, seeds, labels, return_argsort = True)[1]

    dissims = [np.array([footrule(order[k], U[1][i]) + 1 for k in range(K)]) for i in range(len(unlabeled))]
    sims = [np.array([1 - (dissims[i][k]/np.sum(dissims[i][:])) for k in range(K)]) for i in range(len(unlabeled))]

    final_labels = np.zeros(n)
    for i in range(len(unlabeled)):
        winner = np.argwhere(sims[i] == np.max(sims[i]))
        if len(winner) > 1:
            n = len(winner)
            U = uniform(0, 1)
            j = 1
            GreaterThan = False
            while GreaterThan is False:
                if n*U > n - j:
                    GreaterThan = True
                    break
                j += 1
            final_labels[unlabeled[i]] = int(winner[-j][0])
        else:
            final_labels[unlabeled[i]] = int(winner[0][0])
    
    for i in range(len(seeds)):
            final_labels[seeds[i]] = int(labels[i])

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
        
    new_pi = (pis_ * values_) / (pis_ @ values_)

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

def footrule(A, B):
    perm1 = np.array(A)
    perm2 = np.array(B)
    
    K = max(len(perm1), len(perm2))
    
    if K == 0 or K == 1:
        return
    
    zero = np.zeros(K)
    
    if sum(perm1 == zero) == K or sum(perm2 == zero) == K:
        return 0

    c = 0
    for i in range(len(A)):
        c += abs(A[i] - B[i])

    return c

def mc_simulations(n, seed_ratio, pi, B, max_dim, means, scales, it, acorn = 1234):
    np.random.seed(acorn)
    n1 = int(np.round(n*seed_ratio*pi[0]))
    n2 = int(np.round(n*seed_ratio*(1 - pi[0])))

    seeds1 = np.arange(0, n1)
    seeds2 = np.arange(int(np.round(n*pi[0])), int(np.round(n*pi[0])) + n2)
    all_seeds = np.concatenate((seeds1, seeds2))

    labels1 = np.zeros(len(seeds1))
    labels2 = np.ones(len(seeds2))
    seed_labels = np.concatenate((labels1, labels2))

    all_labels = np.concatenate((np.zeros(int(np.round(n*pi[0]))), np.ones(int(np.round(n*pi[1])))))
    unlabeled = [i for i in range(n) if i not in all_seeds]
    
    AOL = []
    LOA = []
    JOINT = []
    AC = []
    PTR = []
    EWC = []
    UP = []
    R = []

    for i in tqdm(range(it)):
        A = adj_matrix(n, pi, B, True, dist = 'normal', means = means, scales = scales, acorn = acorn + i)
        
        A_unweighted = strip_weights(A)
        A_ptr = pass_to_ranks(A)
        
        V, U = adj_spectral_embedding(A_unweighted, max_dim = max_dim)
        V_ptr, U_ptr = adj_spectral_embedding(A_ptr, max_dim = max_dim)
        
        X = V @ np.diag(U)**(1/2)
        X_ptr = V @ np.diag(U)**(1/2)
        
        try:
            AOL.append(np.sum(all_labels == weighted_gaussian_classification(A, all_seeds, seed_labels, model = "gaussian", method = "AOL", X_hat = X))/n)
            LOA.append(np.sum(all_labels == weighted_gaussian_classification(A, all_seeds, seed_labels, model = "gaussian", method = "LOA", X_hat = X))/n)
            JOINT.append(np.sum(all_labels == weighted_gaussian_classification(A, all_seeds, seed_labels, model = "gaussian", method = "joint", X_hat = X))/n)
            AC.append(np.sum(all_labels == gaussian_classification(X, all_seeds, seed_labels))/n)
            PTR.append(np.sum(all_labels == gaussian_classification(X_ptr, all_seeds, seed_labels))/n)
            EWC.append(np.sum(all_labels == edge_weight_classification(A, all_seeds, seed_labels))/n)
            UP.append(np.sum(all_labels == updated_priors_classification(A, all_seeds, seed_labels, pi, max_dim = 1, acorn = acorn + i, X_hat = X))/n)
            R.append(np.sum(all_labels == rank_classification(A, all_seeds, seed_labels, order = ""))/n)
        except:
            print("iteration %i degenerate for $n = %i" %(i, n))
        
    return AOL, LOA, JOINT, AC, PTR, EWC, UP, R

def weighted_gaussian_classification(A, seeds, labels, model = "gaussian", cov_type = "full", params = None, ps = None, method = "", X_hat = ""):
    from math import log

    if A.ndim == 2:
        if A.shape[0] != A.shape[1]:
            raise ValueError("Square matrices only")
    else:
        raise ValueError("2 dimensional arrays only")

    n = A.shape[0]

    if method == "joint":
        edge_likes = weight_likelihoods(A, seeds, labels, model, cov_type, params, ps, method)
        final_labels = np.zeros(n)
        for i in range(n):
            final_labels[i] = np.argmax(edge_likes[i, :])
        for i in range(len(seeds)):
            final_labels[int(seeds[i])] = int(labels[i])
        return final_labels

    B = strip_weights(A)
    
    if type(X_hat) is str:
        X, V = adj_spectral_embedding(B, max_dim = int(round(log(n))))
        elbows = profile_likelihood(V, n_elbows = 2)
        X_hat = X[:, :elbows[1]] @ np.diag(V[:elbows[-1]])**(1/2)

    embed_likes = gaussian_classification(X_hat, seeds, labels, return_likelihoods = True)
    edge_likes = weight_likelihoods(A, seeds, labels, model, cov_type, params, ps = ps, method = method)

    final_labels = np.zeros(n)
    for i in range(n):
        temp_likelihood_product = edge_likes[i, :]*embed_likes[i, :]
        final_labels[i] = np.argmax(temp_likelihood_product)
    for i in range(len(seeds)):
        final_labels[int(seeds[i])] = int(labels[i])

    return final_labels

def weight_likelihoods(A, seeds, labels, model = "gaussian", cov_type = "", params = None, ps = None, method = ""):
    if params is None: # only estimate if the distributions are unknown.. 
        params = estimate_dist(A, seeds, labels, model = model, cov_type = cov_type)  

    n = A.shape[0]

    unique_labels = np.unique(labels)
    K = len(unique_labels)

    new_labels = np.zeros(len(labels))
    for i in range(len(labels)): # reindex labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        new_labels[i] = itemindex

    seed_split = [[] for i in range(K)]
    for i in range(len(seeds)):
        seed_split[int(new_labels[i])].append(seeds[i])

    seed_split = [np.array(seed_split[i]) for i in range(K)]
    nseeds = list(map(len, seed_split))
    max_seeds = max(nseeds)

    likelihoods = np.ones(shape = (n, K))

    if method == "AOL": # average of likelihoods or likelihood of average
        for i in range(n):
            for j in range(K):
                temp_likelihood = np.ones(K)
                for k in range(K):
                    temp_edges = A[i, seed_split[k]][np.nonzero(A[i, seed_split[k]])[0]]
                    if len(temp_edges) != 0:
                        temp = np.array(list(map(lambda weight: norm.pdf(weight, params[0][j][k], params[1][j][k]), temp_edges))) # array of the likelihoods of edge weights
                        temp_likelihood[k] = np.prod(temp)**(1 / len(temp_edges)) # geometric average of likelihoods
                    else:
                        temp_likelihood = np.ones(K) # resets all likelihoods to one if the number of samples == 0
                        break
                likelihoods[i, j] = np.prod(temp_likelihood) # product of the average likelihood
    elif method == "LOA":
        for i in range(n):
            for j in range(K):
                temp_likelihood = np.ones(K)
                for k in range(K):
                    temp_edges = np.array(A[i, seed_split[k]][np.nonzero(A[i, seed_split[k]])[0]]) 
                    if len(temp_edges) != 0:
                        temp_likelihood[k] = norm.pdf(np.mean(temp_edges), params[0][j][k], params[1][j][k]/np.sqrt(len(temp_edges))) # array of the likelihoods of edge weights
                    else:
                        temp_likelihood = np.ones(K) # resets all likelihoods to one if the number of samples == 0
                        break
                likelihoods[i, j] = np.prod(temp_likelihood)
    elif method == "joint" or method == "joint2":
        if ps is None:
            ps = estimate_p(A, seed_split)

        for i in range(n):
            for j in range(K):
                temp_likelihood = np.ones(K)
                for k in range(K):
                    temp_edges = np.array(A[i, seed_split[k]][np.nonzero(A[i, seed_split[k]])[0]])
                    tempk = len(temp_edges)
                    if tempk != 0:
                        temp = np.array(list(map(lambda weight: norm.pdf(weight, params[0][j][k], params[1][j][k]), temp_edges)))
                        temp *= ps[j, k]**(tempk) * (1 - ps[j, k])**(nseeds[k] - tempk)
                        temp_likelihood = np.prod(temp)
                    else:
                        temp_likelihood = np.ones(K)
                        break
                likelihoods[i, j] = np.prod(temp_likelihood)
    return likelihoods

    """
    for i in range(n):
        for j in range(K):
            temp_likelihood = np.ones(K)
            if method == "AOL": # average of likelihoods or likelihood of average
                for k in range(K):
                    temp_edges = A[i, seed_split[k]][np.nonzero(A[i, seed_split[k]])[0]]
                    if len(temp_edges) != 0:
                        temp = np.array(list(map(lambda weight: norm.pdf(weight, params[j][k][0], params[j][k][1]), temp_edges))) # array of the likelihoods of edge weights
                        temp_likelihood[k] = np.prod(temp)**(1 / len(temp_edges)) # geometric average of likelihoods
                    else:
                        temp_likelihood = np.ones(K) # resets all likelihoods to one if the number of samples == 0
                        break
                likelihoods[i, j] = np.prod(temp_likelihood) # product of the average likelihood
            elif method == "LOA":
                for k in range(K):
                    temp_edges = np.array(A[i, seed_split[k]][np.nonzero(A[i, seed_split[k]])[0]])
                    if len(temp_edges) != 0:
                        temp_likelihood[k] = norm.pdf(np.mean(temp_edges), params[j][k][0], params[j][k][1]/np.sqrt(len(temp_edges))) # array of the likelihoods of edge weights
                    else:
                        temp_likelihood = np.ones(K) # resets all likelihoods to one if the number of samples == 0
                        break
                likelihoods[i, j] = np.prod(temp_likelihood)
            elif method == "joint":
                ps = estimate_p(A, seeds, labels)
                for i in range(K):
                    temp_edges = np.array(A[i, seed_split[k]][np.nonzero(A[i, seed_split[k]])[0]])
                    if len(temp_edges) != 0:
                        temp_likelihood = ps[]
    """

def estimate_p(A, seed_splits):
    K = len(seed_splits)
    nseeds = list(map(len, seed_splits))
    max_seeds = max(nseeds)

    ps = np.zeros(shape = (K,K))

    for i in range(K):
        for j in range(nseeds[i]):
            temp_nedges = np.array([len(A[seed_splits[i][j], seed_splits[k]][np.nonzero(A[seed_splits[i][j], seed_splits[k]])[0]]) for k in range(K)])
            ps[i] += temp_nedges

    for i in range(K):
        for j in range(i, K):
            if i == j:
                ps[i,j] = ps[i,j]/(2 * choose(nseeds[i], 2))
            else:
                ps[i,j] = ps[i,j]/(nseeds[i]*nseeds[j])
            ps[j,i] = ps[i,j]
    return ps

def edge_weight_classification(A, seeds, labels, model = "gaussian", cov_type = "", params = "", method = "", acorn = 123):
    likelihoods = weight_likelihoods(A, seeds, labels, model, cov_type, params, method)
    n = A.shape[0]
    final_labels = np.zeros(n)
    for i in range(n):
        winner = np.argwhere(likelihoods[i] == np.amax(likelihoods[i]))
        if len(winner) > 1:
            n = len(winner)
            U = uniform(0, 1)
            j = 1
            GreaterThan = False
            while GreaterThan is False:
                if n*U > n - j:
                    GreaterThan = True
                    break
                j += 1
            final_labels[i] = int(winner[-j][0])
        else:
            final_labels[i] = int(winner[0][0])
    for i in range(len(seeds)):
            final_labels[seeds[i]] = int(labels[i])
    return final_labels

def estimate_dist(A, seeds, labels, model = "gaussian", cov_type = ""):
    # Returns a tuple of the parameters for each distribution in the model

    if len(labels) == 0:
        raise ValueError("No labels")
    elif len(seeds) != len(labels):
        raise ValueError("Unequal number of seeds and labels")

    unique_labels = np.unique(labels)
    K = len(unique_labels)
    NUM_DIST = choose(K, 2) + K
    #print(NUM_DIST)

    new_labels = np.zeros(len(labels))

    for i in range(len(labels)): # reindex labels to [0,.., K-1]
        itemindex = np.where(unique_labels==labels[i])[0][0]
        new_labels[i] = int(itemindex)

    if model == "gaussian":
        weights = [[[] for j in range(K - i)] for i in range(K)]
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                min_ = int(min(new_labels[i], new_labels[j]))
                max_ = int(max(new_labels[i], new_labels[j]))
                diff = max_ - min_
                if A[seeds[i], seeds[j]] != 0:
                    weights[min_][diff].append(A[seeds[i], seeds[j]])

        degenerate = False
        for i in range(K):
            for j in range(K - i):
                if len(weights[i][j]) < 2:
                    degenerate = True
                    deg_i = i + 1
                    deg_j = j + i + 1
                    break

        if degenerate:
            raise ValueError("Less than 2 edges between the labeled nodes of block %i and the labeled nodes of block %i" %(deg_i, deg_j))

        means = np.zeros(shape = (K,K))
        var = np.zeros(shape = (K,K))

        for i in range(K):
            for j in range(i, K):
                temp = np.array(weights[i][i - j])
                means[i,j] = np.mean(temp)
                means[j, i] = means[i,j]

                var[i, j] = np.var(temp, ddof = 1)
                var[j, i] = var[i, j]

        if cov_type == "tied":
            temp_sum = 0
            nedges = 0
            for i in range(K):
                for j in range(i, K):
                    temp_len = len(weights[i][i - j])
                    nedges += temp_len
                    temp_sum += var[i,j]*(temp_len - 1)

            temp_sum = temp_sum / (nedges - NUM_DIST)
            var = temp_sum*np.ones(shape = (K, K))

        params = means, var**(1/2)
        return params
    else:
        raise ValueError("Unsupported model")