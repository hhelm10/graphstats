from __future__ import division
from __future__ import print_function

#- standard imports
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import scipy.sparse as sp

#- statistical inference on graphs
from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.simulations import sbm

#- posterior estimation
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

#- pdf evaluations
from scipy.stats import multivariate_normal as mvn
from scipy.stats import beta, norm

#- module, functions for gcns
import torch
import torch.nn.functional as F
import torch.optim as optim

#- gcn implementation
from pygcn.utils import accuracy
from pygcn.models import GCN as GraphConvolutionalNeuralNetwork


def MVN_sampler(counts, params, seed=None):
    """
    A function to sample from normal distributions.

    counts - The number of samples to generate from each distribution.
    params - a K-list of parameters where params[k][0] is the mean for the kth
        Gaussian and params[k][1] is the covariance for the kth Gaussian if p > 1
        and the corresponding standard deviation if p == 1.
    seed - random generator seed
    """
    if seed is None:
        seed = np.random.randint(10**6)
    np.random.seed(seed)
    
    n = np.sum(counts)
    K = len(params)

    if isinstance(params[0][0], np.float64):
        p = 1
    else:
        p = len(params[0][0])

    if p == 1:
        norm_sampler = np.random.normal
        samples = np.zeros(n)
        if not isinstance(params[0][1], np.float64):
            for k in range(K):
                params[k][1] = params[k][1][0]
    else:
        norm_sampler = np.random.multivariate_normal
        samples = np.zeros((n,p))
    
    for i in range(K):
        idx = np.arange(sum(counts[:i]),np.sum(counts[:i+1]))
        samples[idx] = norm_sampler(params[i][0], params[i][1], counts[i])
        
    return samples


def beta_sampler(counts, params, seed=None):
    if seed is None:
        seed = np.random.randint(10**6)
    np.random.seed(seed)
    
    n = np.sum(counts)
    d = len(params[0])
    K = len(counts)
        
    samples = np.zeros((n, d))
    for k in range(K):
        for i in range(d):
            samples[np.sum(counts[:k]): np.sum(counts[:k+1]), i] = np.random.beta(params[k][0][i],
                                                                                  params[k][1][i], 
                                                                                  counts[k])
    return samples


def conditional_MVN_sampler(Z, rho, counts, params, seed=None):
    # could be written in MVN_sampler
    # rho is not 'correlation' but just a scale for Z_{ij}. however, rho = 0 => correlation = 0
    if seed is None:
        seed = np.random.randint(10**6)
        
    np.random.seed(seed)
    K = len(counts)
    
    X = np.zeros(Z.shape)
    p = len(params[0][0])

    if p == 1:
        norm_sampler = np.random.normal
    
    for i in range(K):
        for c in range(counts[i]):
            temp_Z = np.prod(Z[np.sum(counts[:i]) + c])
            X[np.sum(counts[:i]) + c] = rho*temp_Z*np.random.multivariate_normal(params[i][0], 
                                                                      params[i][1])
    
    return X


def estimate_normal_parameters(samples):
    """
    Simple function to estiamte normal parameters.
    """

    if len(samples[0]) is 1:
        samples = samples.reshape((1, -1))[0]
        return np.mean(samples), np.std(samples, ddof=1)
    elif samples.ndim is 2:
        return np.mean(samples, axis = 0), np.cov(samples, rowvar=False)
    else:
        raise ValueError('tensors not supported')


def classify(X, Z, normal_params, fitted_model, m = None):
    """
    Classifies vertices.

    X - n x p; Normally distributed random variables.
    Z - n x d; Not normally distributed random variables.
    fitted_model - A sklearn model that can return posterior estimates.
    m - number of training data used to train fitted_model.
    """

    n, p = X.shape
    m, d = Z.shape
    
    K = len(normal_params)
    
    if n != m:
        raise ValueError('different number of samples for X, Z')
    
    if p == 1:
        norm_pdf = norm.pdf
        X = X.reshape((1, -1))[0]
    else:
        norm_pdf = mvn.pdf
        
    posteriors = fitted_model.predict_proba(Z)
    
    predictions=-1*np.zeros(n)

    for i in range(n):
        if m is None:
            smoothed_posterior = posteriors[i]
        else:
            posterior_plus = posteriors[i] + np.ones(K)/m
            smoothed_posterior = posterior_plus / np.sum(posterior_plus)
        temp_pdfs = np.array([norm_pdf(X[i], normal_params[j][0], normal_params[j][1]) for j in range(K)])
        posterior_pdf_prod = temp_pdfs * smoothed_posterior
        predictions[i] = int(np.argmax(posterior_pdf_prod))
        
    return predictions


def error_rate(truth, predictions, seed_idx = None, metric = 'accuracy'):
    """
    If metric is 'accuracy', calculates 0-1 loss.
    """
    if metric == 'accuracy':
        if seed_idx is None:
            return 1 - np.sum(predictions == truth)/len(truth)
        

def estimate_bayes(n, pi, normal_params, beta_params, seed=None):
    if seed is None:
        seed = np.random.randint(10**6)
        
    p = len(normal_params[0][0])
    if p == 1:
        norm_pdf = norm.pdf
    else:
        norm_pdf = mvn.pdf
        
    K = len(normal_params)
    
    d = len(beta_params[0])
        
    counts = np.round(np.array(n*(pi*np.ones(K))).astype(int))
    labels = np.concatenate((np.zeros(counts[0]), np.ones(counts[1])))
    X = MVN_sampler(counts, normal_params)
    Z = beta_sampler(counts, beta_params)
    
    predictions = -1*np.ones(n)
    for i in range(n):
        normal_pdfs = [pi[k]*norm_pdf(X[i], normal_params[0][k], normal_params[1][k]) for k in range(K)]
        beta_pdfs = np.ones(d)
        for j in range(d):
            for k in range(K):
                beta.pdf(Z[i,j], beta_params[k][0][j], beta_params[k][1][j])
            beta_pdfs *= np.array([beta.pdf(Z[i,j], beta_params[k][0][j], beta_params[k][1][j]) for k in range(K)])
        predictions[i] = np.argmax(normal_pdfs * beta_pdfs)
        
    return error_rate(labels, predictions)


def blowup(P, tau):
    """
    Turns a K x K probability matrix into a len(tau) x len(tau) 
    or np.sum(tau) x np.sum(tau).

    tau - an array or list of class labels or an array or list of class counts.
    """
    unique_labels = np.unique(tau)

    # There are two reasonable things to pass this function
    # 1) A vector of counts where tau[i] is the number of nodes in block i
    # 2) A vector of class labels where tau[i] is the class label for node i

    if len(unique_labels) == len(tau):
        n = np.sum(tau)
        new_tau = np.concatenate([i*np.ones(tau[i]) for i in range(len(tau))]).astype(int)
    else:
        n = len(tau)
        new_tau = -1*np.ones(n).astype(int)

        for i in range(n):
            new_tau[i] = int(np.where(tau[i]==unique_labels)[0][0])

    blown_up = np.zeros((n, n))
    
    for i in range(n):
        temp_label1 = new_tau[i]
        for j in range(i+1, n):
            temp_label2 = new_tau[j]
            blown_up[i,j] = P[new_tau[i], new_tau[j]]
            blown_up[j,i] = P[new_tau[j], new_tau[i]]
            
    return blown_up


def QDA(X, pi, params):
    """ 
    An implementation of quadratic discriminant analysis.

    X - n_samples x n_features 
    pi - element of K-1 dimensional simplex
    params - a K-list of parameters where params[k][0] is the mean for the kth
        Gaussian and params[k][1] is the covariance for the kth Gaussian if p > 1
        and the corresponding standard deviation if p == 1.
    """
    n, p = X.shape
    
    K = len(params)
    
    if p == 1:
        norm_pdf = norm.pdf
    else:
        norm_pdf = mvn.pdf
    
    predictions=-1*np.zeros(n)
    for i in range(n):
        temp_pdfs = np.array([norm_pdf(X[i], params[j][0], params[j][1]) for j in range(K)])
        predictions[i] = int(np.argmax(temp_pdfs))
        
    return predictions


def rank1_variance(pi, p, q):
    """
    A function that finds the theoretical variance for the 2 block, rank 1
    positive definite Stochastic Block model.
    B = [
        [p**2, p*q],
        [p*q, q**2]
    ]

    pi - element of 1 dimenesional simplex
    """

    pi0 = pi
    pi1 = 1 - pi0
    
    num11 = pi0 * p**4 * (1 - p**2)
    num12 = pi1 * p * q**3 * (1 - p*q)
    num1 = num11 + num12
    
    num21 = pi0* p**3 * q * (1 - p*q)
    num22 = pi1 * p * q**3 * (1 - p*q)
    num2 = num21 + num22
    
    den = (pi0* p**2 + pi1 * q**2)**2
    return [num1/den, num2/den]


def GCN(adj, features, train_idx, labels, epochs=200, n_hidden=16, dropout=0.5, learning_rate=0.01, weight_decay=5e-4, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)

    model = GraphConvolutionalNeuralNetwork(nfeat=features.shape[1],
            n_hidden=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)
    optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[train_idx], labels[train_idx])
        acc_train = accuracy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

    test_idx = np.array([i for i in range(adj.shape[0]) if i not in train_idx])

    def test():
        model.eval()
        output = model(features, adj)
        # loss_test = F.nll_loss(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])

        return acc_test

    for epoch in range(epochs):
        train(epoch)

    return test()


def simulation(n, pi, normal_params, beta_params, cond_ind=True, errors = None, smooth=False, acorn=None):
    #- Type checks
    if isinstance(normal_params, list):
        sbm_check = False
        # there are other checks to do..
    elif isinstance(normal_params, np.ndarray):
        if normal_params.ndim is 2:
            if np.sum(normal_params == normal_params.T) == np.prod(normal_params.shape):
                sbm_check = True
            else:
                msg = 'if normal_params is a 2 dimensional array it must be symmetric'
                raise ValueError(msg)
        else:
            msg = 'if normal_params is an array, it must be a 2 dimensional array'
            raise TypeError(msg)
    else:
        msg = 'normal_params must be either a list or a 2 dimensional array'
        raise TypeError(msg)

    if acorn is None:
        acorn = np.random.randint(10**6)
    np.random.seed(acorn)

    #- Multinomial trial
    counts = np.random.multinomial(n, [pi, 1 - pi])

    #- Hard code the number of blocks
    K = 2

    #- Set labels
    labels = np.concatenate((np.zeros(counts[0]), np.ones(counts[1])))

    #- number of seeds = n_{i}/10
    n_seeds = np.round(0.1*counts).astype(int)

    #- Set training and test data
    class_train_idx = [range(np.sum(counts[:k]), np.sum(counts[:k]) + n_seeds[k]) for k in range(K)]
    train_idx = np.concatenate((class_train_idx)).astype(int)

    test_idx = [k for k in range(n) if k not in train_idx]

    #- Total number of seeds
    m = np.sum(n_seeds)

    #- estimate class probabilities
    pi_hats = n_seeds / m

    #- Sample from beta distributions
    beta_samples = beta_sampler(counts, beta_params)
    Z = beta_samples

    #- Sample from multivariate normal or SBM either independently of Zs or otherwise
    if cond_ind:
        if sbm_check:
            A = sbm(counts, normal_params)
            ase_obj = ASE(n_elbows=1)
            X = ase_obj.fit_transform(A)
        else:
            X = MVN_sampler(counts, normal_params)
            if len(normal_params[0][0]) is 1:
                X = X[:, np.newaxis]
    else:
        if sbm_check:
            P = blowup(normal_params, counts) # A big version of B to be able to change connectivity probabilities of individual nodes
            scales = np.prod(Z, axis=1)**(1/Z.shape[1]) # would do just the outer product, but if the Z's are too small we risk not being connected 
            new_P = P*(scales @ scale.T) # new probability matrix
            A = sbm(np.ones(n).astype(int), new_P) 
            ase_obj = ASE(n_elbows=1)
            X = ase_obj.fit_transform(A)
        else:
            X = conditional_MVN_sampler(Z=Z, rho=1, counts=counts, params=normal_params, seed=None)
            if len(normal_params[0][0]) is 1:
                X = X[:, np.newaxis]

    XZ = np.concatenate((X, Z), axis=1)

    #- Estimate normal parameters using seeds
    params = []
    for i in range(K):
        temp_mu, temp_cov = estimate_normal_parameters(X[class_train_idx[i]])
        params.append([temp_mu, temp_cov])

    #- Using conditional indendence assumption (RF, KNN used for posterior estimates)
    if errors is None:
        errors = [[] for i in range(5)]

    rf1 = RF(n_estimators=100, max_depth=int(np.round(np.log(Z[train_idx].shape[0]))))
    rf1.fit(Z[train_idx], labels[train_idx])

    knn1 = KNN(n_neighbors=int(np.round(np.log(Z[train_idx].shape[0]))))
    knn1.fit(Z[train_idx], labels[train_idx])

    if smooth:
        temp_pred = classify(X[test_idx], Z[test_idx], params, rf1, m = m)
        temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
        errors[0].append(temp_error)

        temp_pred = classify(X[test_idx], Z[test_idx], params, knn1, m = m)
        temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
        errors[1].append(temp_error)
    else:
        temp_pred = classify(X[test_idx], Z[test_idx], params, rf1)
        temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
        errors[0].append(temp_error)

        knn1 = KNN(n_neighbors=int(np.round(np.log(m))))
        knn1.fit(Z[train_idx], labels[train_idx])

        temp_pred = classify(X[test_idx], Z[test_idx], params, knn1)
        temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
        errors[`].append(temp_error)

    temp_pred = QDA(X[test_idx], pi_hats, params)
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[1].append(temp_error)

    #- Not using conditional independence assumption (RF, KNN used for classification)
    XZseeds = np.concatenate((X[train_idx], Z[train_idx]), axis=1)

    rf2 = RF(n_estimators=100, max_depth=int(np.round(np.log(m))))
    rf2.fit(XZ[train_idx], labels[train_idx])
    temp_pred = rf2.predict(XZ[test_idx])
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[3].append(temp_error)

    knn2 = KNN(n_neighbors=int(np.round(np.log(m))))
    knn2.fit(XZ[train_idx], labels[train_idx])

    temp_pred = knn2.predict(XZ[test_idx])
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[4].append(temp_error)

    temp_accuracy = GCN(adj, features, train_idx, labels)
    temp_error = 1 - temp_accuracy
    errors[5].append(temp_error)

    return errors

def plot_errors(sample_sizes, errors, labels, xlabel=None, ylabel=None, title=None, png_title=None, bayes=None):
    n_n_samples = len(errors)
    n_classifiers = len(errors[0])
    if n_n_samples != len(sample_sizes):
        msg = 'invalid list of errors. each list must be the same length as {}'.format(sample_sizes)
        raise ValueError(msg)

    # Probably a way to do this more efficiently
    means = [np.zeros(n_n_samples) for _ in range(n_classifiers)]
    stds = [np.zeros(n_n_samples) for _ in range(n_classifiers)]
    for i, n in enumerate(sample_sizes):
        for j in range(n_classifiers):
            means[j][i] = np.mean(errors[i][j])
            stds[j][i] = np.std(errors[i][j], ddof=1)/np.sqrt(len(errors[i][j]))

    fig, ax = plt.subplots(1,1)
    sns.set(palette=sns.color_palette("Set1", n_colors = len(labels)))
    for i in range(n_classifiers):
        ax.plot(sample_sizes, means[i], label=labels[i])
        ax.fill_between(sample_sizes, 
            means[i] + 1.96*stds[i], 
            means[i] - 1.96*stds[i], 
            where=means[i] + 1.96*stds[i] >= means[i] - 1.96*stds[i], 
            facecolor=colors[i], 
            alpha=0.15,
            interpolate=True)

    if bayes is not None:
        ax.plot(sample_sizes, bayes, label='oracle bayes')

    if xlabel is None:
        xlabel = 'n'
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = 'misclassification rate'
    ax.set_ylabel(ylabel)

    if title is None:
        title = 'misclassification rate vs n'
    ax.set_title(title)

    ax.legend()

    if png_title is not None:
        plt.savefig(png_title + '.png')

    return