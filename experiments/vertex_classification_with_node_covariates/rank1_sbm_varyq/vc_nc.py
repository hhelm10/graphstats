from covariates_gclass import *
# import numpy as np
# import matplotlib.pyplot as plt
# import graspy
# from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from scipy.stats import multivariate_normal as mvn, beta, norm
from tqdm import tqdm as tqdm
# from graspy.embed import AdjacencySpectralEmbed as ASE
# from graspy.simulations import sbm
import seaborn as sns
sns.set()
import _pickle as pickle

np.random.seed(2)

pi0 = 0.5
seed_ratio = 0.1

p = 0.6
q = 0.5

std1, std2 = np.sqrt(rank1_variance(pi0, p, q))

w = 1
mean = 0.05*np.ones(w)
cov = np.eye(w)

normal1 = [mean, std1*cov]
normal2 = [-mean, std2*cov]
normal_params = [normal1, normal2]

d = 2
a1 = 4*np.ones(d)
a2 = 6*np.ones(d)
beta1 = [a1, a2] # beta1 = [[a1, a2, a3, .., ad], [b1, b2, .., bd]]
beta2 = [a2, a1]
beta_params = [beta1, beta2]

# M = 200

n = [75, 100, 150, 200, 300, 400, 500] #, 2000] #, 5000]

# all_errors_norm = []

# print("begin true normal (simulaion set 1)")
# for i in tqdm(range(len(n))):
#     errors_norm = [[] for i in range(5)]
#     for j in range(M):
#         temp_norm = simulation(n[i], 0.5, normal_params, beta_params, cond_ind=True, errors=errors_norm, smooth=True)
#         errors_norm = temp_norm
        
#     all_errors_norm.append(errors_norm)

# plot_errors(n, all_errors_norm, labels = ['qda', 'hhrf', 'hhknn', 'rf', 'knn'], png_title = 'true_normal')
# pickle.dump(all_errors_norm, open("true_normal_errors_20191804.pkl", 'wb'))

# print("done true normal (simulation set 1)")


#- 2block rank1 SBM
print("begin rank 1 sbms (simulation set 2)")

qs = np.arange(0.35, 0.6, step=0.05)
print(qs)

M = 200

n = [75, 100, 150, 200, 300, 400, 500]
for k, q in enumerate(tqdm(qs)):
    print(k, q)
    B = np.array([
        [p**2, p*q],
        [p*q, q**2]
    ])

    all_errors_sbm = []

    for i in range(len(n)):
        errors_sbm = [[] for i in range(5)]
        for j in range(M):
            try:
                temp_sbm = simulation(n[i], 0.5, B, beta_params, cond_ind=True, errors=errors_sbm, smooth=True)
                errors_sbm = temp_sbm
            except:
                print('fail')
                pass
        
        all_errors_sbm.append(errors_sbm)

    plot_errors(n, all_errors_sbm, labels = ['qda', 'hhrf', 'hhknn', 'rf', 'knn'], png_title = 'sbm_rank1_p6_q%i'%(int(100*q)))
    pickle.dump(all_errors_sbm, open('all_errors_sbm_rank1_p6_q%i_20191904.pkl'%(int(100*q)), 'wb'))

print("done rank1 sbms (simulation set 2)")

#- 2block rank2 SBM


#- 2block rank2 SBM

# p = 0.6
# qs = np.arange(3, 6, step=0.05)/10

# std1, std2 = np.sqrt(rank1_variance(pi0, p, 0.55))

# w = 1
# mean = 0.025*np.ones(w)
# cov = np.eye(w)

# normal1 = [mean, std1*cov]
# normal2 = [-mean, std2*cov]
# normal_params = [normal1, normal2]

# np.random.seed(2)

# M = 100

# n = [75, 100, 150, 200, 300, 400, 500] #, 2000] #, 5000]

# all_errors_norm = []
# all_errors_sbm = []

# for i in tqdm(range(len(n))):
#     errors_norm = [[] for i in range(5)]
#     errors_sbm = [[] for i in range(5)]
#     for j in range(M):
#         temp_norm = simulation(n[i], 0.5, normal_params, beta_params, cond_ind=True, errors=errors_norm, smooth=True)
#         errors_norm = temp_norm
        
#         temp_sbm = simulation(n[i], 0.5, B, beta_params, cond_ind=True, errors=errors_sbm, smooth=True)
#         errors_sbm = temp_sbm
        
#     all_errors_norm.append(errors_norm)
#     all_errors_sbm.append(errors_sbm)