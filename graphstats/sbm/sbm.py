import numpy as np

def SBM(n, pi = [], B = [], weighted = False, dist = "", params = [], acorn = 1234):
    if len(B) == 0:
        B = [np.eye(round(n/2))]
    
    if len(pi) == 0:
        pi = 1/B.shape[0] * np.ones(B.shape[0])
   
    K, _ = B.shape
    A = adj_matrix(n, pi, B, weighted = weighted, dist = dist, params = params, acorn = acorn)

    return A

def gen_pi(K = 4, v = 20, acorn = 1234):
    np.random.seed(acorn)
    #Taken from Bijan
    
    while True:
        pi_0 = np.random.uniform(size=K) # Generate K Uniform(0,1) random variables
        pisum = np.sum(pi_0) # Calculate the sum of the K generated Uniform(0,1) random variables
        pi = [i/pisum for i in pi_0] # Normalize so that sum(pi) = 1
        Nv = [int(round(i*v)) for i in pi] # Round so that each block has an integer-valued amount of vertices
        if not 0 in Nv and np.sum(Nv)==v: # Make sure no block has 0 vertices and the sum of all vertices is correct
            break
    return pi, Nv # returns the vertex assignment distribution and the number of vertices in each block

def gen_Lambda(K, acorn=1234, ones_ = False):
    if ones_ == True:
        return np.ones(shape = (K, K))
    
    np.random.seed(acorn)
    Lambda = np.zeros(shape = (K, K)) # K x K matrix to store adjacency probabilities
    for i in range(K): # for each block
        for j in range(i, K): # for each combination (with replacement)
            Lambda[i, j] = np.random.uniform() # generate a Uniform(0,1) random variable
            Lambda[j, i] = Lambda[i, j] # Lambda is symmetric
                   
    return Lambda # returns a K x K, symmetric matrix where Lambda[i,j] in (0, 1)

def gen_F(K, equal = False, dist = "poisson", min_ = 1, max_ = 100, acorn = 1234): # Used for weighted graphs
    np.random.seed(acorn) # set seed

    F = np.ones(K**2) # start with an array of ones for equal case
    if equal: # if all the distributions are equal
    	F = F*np.random.randint(min_, max_)
    	
    F = F.reshape((K, K)) # reshape to a K x K matrix
    for i in range(K): 
        for j in range(i, K):
            F[i,j] = np.random.randint(min_, max_) # randomly set distributions for each block
            F[j, i] = F[i,j] # symmetry!
            
    return F

def adj_matrix(n, pi, Lambda, weighted = False, dist = "", params = [], acorn = 1234):
    np.random.seed(acorn)
    n = int(n) # Just in case!
    A = np.zeros(shape = (n, n)) # n x n adjcacency matrix
    K = len(pi) # extract the number of blocks in the SBM
    
    i = 0 # start at block indexed with 0
    while i < K: # while the block number is less than the total number of blocks
        for k in range(int(round(n*(sum(pi[:i])))), int(round(n*(sum(pi[:i + 1]))))): # for all vertices in block i
            c = i # start at block i
            while c < K: # while the block number is less than the total number of blocks
                for j in range(int(round(n*(sum(pi[:c])))), int(round(n*(sum(pi[:c + 1]))))): # for all vertices in block c
                    A[k, j] = np.random.binomial(1, Lambda[i, c]) # generates and assigns an edge based on block membership
                    if weighted:
                        if dist == "poisson":
                            A[k, j] = A[k, j] * (np.random.poisson(params[i, c]) + 1)
                    A[j, k] = A[k, j] # A is symmetric
                c += 1
            A[k,k] = 0 # A is hollow
        i += 1
        
    return A # returns an n x n, symmetric and hollow matrix where A[i,j] in {0, 1}

def gen_seeds(Nv, seed_ratio, acorn = 1234):
    np.random.seed(acorn)

    K = len(Nv)
    
    num_seeds = [int(round(seed_ratio*Nv)) for i in range(K)]

    seeds = [[] for i in range(K)]

    for i in range(K):
    	for k in range(num_Seeds[i]):
    		seeds[i].append(int(i + k + sum(Nv[:i])))

    return seeds