import numpy as np
import matplotlib.pyplot as plt
import graspy
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.stats import multivariate_normal as mvn, beta, norm
from tqdm import tqdm_notebook as tqdm
from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.simulations import sbm

def MVN_sampler(counts, params, seed=None):
    if seed is None:
        seed = np.random.randint(10**6)
    np.random.seed(seed)
    
    n = np.sum(counts)
    K = len(params)
    p = len(params[0][0])
    
    samples = np.zeros((n,d))
    c = 0
    class_idx = 0
    
    for i in range(K):
        idx = np.arange(sum(counts[:i]),np.sum(counts[:i+1]))
        samples[idx] = np.random.multivariate_normal(params[0][i], params[1][i], counts[i])
        
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
    
    for i in range(K):
        for c in range(counts[i]):
            temp_Z = np.prod(Z[np.sum(counts[:i]) + c])
            X[np.sum(counts[:i]) + c] = rho*temp_Z*np.random.multivariate_normal(params[0][i], 
                                                                      params[1][i])
    
    return X

def estimate_normal_parameters(samples):
    return np.mean(samples, axis = 0), np.cov(samples, rowvar=False)

def classify(X, Z, normal_params, fitted_model):
    n, p = X.shape
    m, d = Z.shape
    
    K = len(normal_params)
    
    if n != m:
        raise ValueError('different number of samples for X, Z')
    
    if p == 1:
        norm_pdf = norm.pdf
    else:
        norm_pdf = mvn.pdf
        
    posteriors = fitted_model.predict_proba(Z)
    
    predictions=-1*np.zeros(n)
    for i in range(n):
        temp_pdfs = np.array([norm_pdf(X[i,:], normal_params[j][0], normal_params[j][1]) for j in range(K)])
        posterior_pdf_prod = temp_pdfs * posteriors[i]
        predictions[i] = int(np.argmax(posterior_pdf_prod))
        
    return predictions

def error_rate(truth, predictions, seed_idx = None, metric = 'accuracy'):
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

def simulation(n, pi, normal_params, beta_params, cond_ind=True, errors = None, acorn=None):
    #- Type checks
    if isinstance(normal_params, list):
        sbm_check = False
        # there are other checks to do..
    elif isinstance(normal_paramsis, np.ndarray):
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
            ase_obj = ASE(n_elbows)
            X = ase.fit_transform(A)
        else:
            X = MVN_sampler(counts, normal_params)
    else:
        if sbm_check:
            msg = 'non conditionally independent sbm not implemented'
            raise ValueError(msg)
        else:
            X = conditional_MVN_sampler(Z=Z, rh0=1, counts=counts, params=normal_params, seed=None)


    XZ = np.concatenate((X, Z), axis = 1)

    #- Store mvn samples corresponding to seeds

    seeds_norm = X[train_idx]

    #- Estimate normal parameters using seeds
    mu1, cov1 = estimate_normal_parameters(X[class_train_idx[0]])
    params1 = [mu1, cov1]

    mu2, cov2 = estimate_normal_parameters(X[class_train_idx[1]])
    params2 = [mu2, cov2]

    #- Convenient way to store
    params=[params1, params2]

    #- Store uniform samples corresponding to seeds
    seeds_beta = Z[train_idx]

    #- Using conditional indendence assumption (RF, KNN used for posterior estimates)
    rf1 = RF(n_estimators=100, max_depth=np.log(seeds_beta.shape[0]))
    rf1.fit(seeds_beta, labels[train_idx])

    if errors is None:
        errors = [[] for i in range(4)]

    temp_pred = classify(X[test_idx], Z[test_idx], params, rf1)
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[0].append(temp_error)

    knn1 = KNN(n_neighbors=int(np.round(np.log(m))))
    knn1.fit(seeds_beta, labels[train_idx])

    temp_pred = classify(X[test_idx], Z[test_idx], params, knn1)
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[1].append(temp_error)

    #- Not using conditional independence assumption (RF, KNN used for classification)
    XZseeds = np.concatenate((seeds_norm, seeds_beta), axis=1)

    rf2 = RF(n_estimators=100, max_depth=np.log(XZseeds.shape[0]))
    rf2.fit(XZseeds, labels[train_idx])
    temp_pred = rf2.predict(XZ[test_idx])
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[2].append(temp_error)

    knn2 = KNN(n_neighbors=int(np.round(np.log(m))))
    knn2.fit(XZseeds, labels[train_idx])

    temp_pred = knn2.predict(XZ[test_idx])
    temp_error = 1 - np.sum(temp_pred == labels[test_idx])/len(test_idx)
    errors[3].append(temp_error)

    return errors

def plot_errors(sample_sizes, errors, labels, bayes=None):
    n_n_samples = len(errors)
    n_classifiers = len(errors[0])
    if n_n_samples != sample_sizes:
        msg = 'invalid list of errors. each list must be the same length as {}'.format(sample_sizes)
        raise ValueError(msg)

    n_mc = len(errors[0][0])

    # Probably a way to do this more efficiently
    means = [np.zeros(n_n_samples) for _ in n_classifiers]
    stds = [np.zeros(n_n_samples) for _ in n_classifiers]
    for i, n in enumerate(sample_sizes):
        for j in range(n_classifiers):
            means[j] = np.mean(errors[i][j])
            stds[j] = np.std(errors[i][j], ddof=1)/np.sqrt(len(errors[i][j]))

    fig, ax = plt.subplots(1,1)
    for i in range(n_classifiers):
        ax.plot(sample_sizes, means[i])
        ax.fill_between(HH_means, 
            means[i] + 1.96*stds[i], 
            means[i] - 1.96*stds[i], 
            where=means[i] + 1.96*stds[i] >= means[i] - 1.96*stds[i], 
            facecolor='b', 
            alpha=0.15,
            interpolate=True)
    return