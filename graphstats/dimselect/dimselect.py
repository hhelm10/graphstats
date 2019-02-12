#!/usr/bin/env python

# dimselect.py
# Copyright (c) 2017. All rights reserved.

import numpy as np
from scipy.stats import norm

def profile_likelihood(L, n_elbows = 1, max_dim = 100, threshold = 0):
    """
    An implementation of profile likelihood as outlined in Zhu and Ghodsi.
    
    Inputs
        L - An ordered or unordered list of eigenvalues
        n - The number of elbows to return
        threshold - Smallest value to consider. Nonzero thresholds will affect elbow selection.

    Return
        elbows - A numpy array containing elbows


    """

    U = L.copy()
    U = U[:max_dim]

    if type(U) == list: # cast to array for functionality later
        U = np.array(U)
    
    if n_elbows == 0: # nothing to do..
        return np.array([])
    
    if U.ndim == 2:
        U = np.std(U, axis = 0)
    
    # select values greater than the threshold
    U = U[U > threshold]
    
    if len(U) == 0:
        return np.array([])
    
    elbows = []
    
    if len(U) == 1:
        return np.array(elbows.append(U[0]))
    
    U.sort() # sort
    U = U[::-1] # reverse array so that it is sorted in descending order
    n = len(U)

    while len(elbows) < n_elbows and len(U) > 1:
        d = 1 # index of the first elbow +1; intended to be used as list[:d]
        sample_var = np.var(U, ddof = 1)
        sample_scale = sample_var**(1/2)
        elbow = 0 # Initialize the elbow
        likelihood_elbow = 0 # Initialize the likelihood given
       
        while d < len(U):
            mean_sig = np.mean(U[:d]) # Mean of the values considered signal
            mean_noise = np.mean(U[d:]) # Mean of the values considered noise
            sig_likelihood = 0 # Initailize signal likelihood
            noise_likelihood = 0 # Initalize noise likelihood
            for i in range(d):
                sig_likelihood += norm.pdf(U[i], mean_sig, sample_scale) # Update signal likelihood
            for i in range(d, len(U)):
                noise_likelihood += norm.pdf(U[i], mean_noise, sample_scale) # Update noise likelihood
            
            likelihood = noise_likelihood + sig_likelihood # likelihood for the current crtical valule
        
            if likelihood > likelihood_elbow: # Update likelihooh if current likelihood is greater than the current max
                likelihood_elbow = likelihood
                elbow = d # new elbow
            d += 1
        if len(elbows) == 0:
            elbows.append(elbow)
        else:
            elbows.append(elbow + elbows[-1])
        U = U[elbow:]
        
    if len(elbows) == n_elbows:
        return np.array(elbows)
    
    if len(U) == 0:
        return np.array(elbows)
    else:
        elbows.append(n)
        return np.array(elbows)

def ZG(L, n_elbows = 1, threshold = 0):
    """
    An implementation of profile likelihood as outlined in Zhu and Ghodsi.
    
    Inputs
        L - An ordered or unordered list of eigenvalues
        n - The number of elbows to return
        threshold - Smallest value to consider. Nonzero thresholds will affect elbow selection.

    Return
        elbows - A numpy array containing elbows


    """

    U = L.copy()

    if type(U) == list: # cast to array for functionality later
        U = np.array(U)
    
    if n_elbows == 0: # nothing to do..
        return np.array([])
    
    if U.ndim == 2:
        U = np.std(U, axis = 0)
    
    # select values greater than the threshold
    U = U[U > threshold]
    
    if len(U) == 0:
        return np.array([])
    
    elbows = []
    
    if len(U) == 1:
        return np.array(elbows.append(U[0]))
    
    U.sort() # sort
    U = U[::-1] # reverse array so that it is sorted in descending order
    n = len(U)

    while len(elbows) < n_elbows and len(U) > 1:
        d = 1 # index of the first elbow +1; intended to be used as list[:d]
        sample_var = np.var(U, ddof = 1)
        sample_scale = sample_var**(1/2)
        elbow = 0 # Initialize the elbow
        likelihood_elbow = 0 # Initialize the likelihood given
        if len(elbows) == 0:
            diff = []
            diff_var = []
        while d < len(U):
            mean_sig = np.mean(U[:d]) # Mean of the values considered signal
            mean_noise = np.mean(U[d:]) # Mean of the values considered noise
            sig_likelihood = 0 # Initailize signal likelihood
            noise_likelihood = 0 # Initalize noise likelihood
            for i in range(d):
                sig_likelihood += norm.pdf(U[i], mean_sig, sample_scale) # Update signal likelihood
            for i in range(d, len(U)):
                noise_likelihood += norm.pdf(U[i], mean_noise, sample_scale) # Update noise likelihood
            
            likelihood = noise_likelihood + sig_likelihood # likelihood for the current crtical valule

            if len(elbows) == 0 and d < len(U) - 1:

                test = 0
                test_var = 0
                test1 = 0
                test1_var = 0
                for i in range(len(U) - 1):
                    if i < d:
                        test += (mean_sig - U[i])**2/sample_scale
                        test_var += (mean_sig - U[i])**2/sample_var
                    else:
                        test1 +=  (mean_noise - U[i])**2/sample_scale
                        test1_var += (mean_noise - U[i])**2/sample_var
                temp_diff = test - test1
                temp_diff_var = test_var - test1_var
                
                
                diff.append(temp_diff)
                diff_var.append(temp_diff_var)
                print(d, test / test1, test_var / test1_var)
        
            if likelihood > likelihood_elbow: # Update likelihooh if current likelihood is greater than the current max
                likelihood_elbow = likelihood
                elbow = d # new elbow
            d += 1
        if len(elbows) == 0:
            elbows.append(elbow)
        else:
            elbows.append(elbow + elbows[-1])
        U = U[elbow:]
        
    if len(elbows) == n_elbows:
        return np.array(elbows), diff, diff_var
    
    if len(U) == 0:
        return np.array(elbows), diff, diff_var
    else:
        elbows.append(n)
        return np.array(elbows), diff, diff_var