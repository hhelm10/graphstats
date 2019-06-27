from covariates_gclass import *
from tqdm import tqdm as tqdm
import seaborn as sns
sns.set()
import _pickle as pickle
import time

np.random.seed(2)

pi0 = 0.5
seed_ratio = 0.1

p = 0.6
q = 0.3

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

n = [75, 100, 150, 200, 300, 400, 500]

#- 2block rank1 SBM
print("begin rank 1 sbms (simulation set 1)")

alphas = np.concatenate((np.arange(4, 6.5, step=0.5), [7]))
print(alphas)

M = 2
B = np.array([
        [p**2, p*q],
        [p*q, q**2]
])

n = [75, 100, 150, 200, 300, 400, 500]

a1 = 4*np.ones(d)
a2 = 5*np.ones(d)
beta1 = [a1, a2] # beta1 = [[a1, a2, a3, .., ad], [b1, b2, .., bd]]
# beta_params = [beta1, beta2]
for k, alpha in enumerate(tqdm(alphas)):
    temp_a2 = alpha * np.ones(d)
    beta2 = [temp_a2, a1]
    beta_params = [beta1, beta2]
    

    all_errors_sbm = []

    for i in range(len(n)):
        errors_sbm = [[] for i in range(6)]
        for j in range(M):
            try:
                temp_sbm = simulation(n[i], 0.5, B, beta_params, cond_ind=True, errors=errors_sbm, smooth=True)
                errors_sbm = temp_sbm
                failed = False
            except Exception as e:
                print(str(e))
                print('fail, %i, %1.2f'%(k, alpha))
                failed = True
            
            if failed:
                time.sleep(5)

        all_errors_sbm.append(errors_sbm)

    plot_errors(n, 
    	all_errors_sbm, 
    	labels = ['SDA-RF', 'SDA-kNN', 'QDA', 'RF', 'kNN', 'GCN'], 
    	xlabel = 'n',
    	ylabel = 'Misclassification Rate',
    	title = 'Misclassification Rate vs n',
    	png_title = 'sbm_rank1_p6_alpha%i'%(int(10*alpha)))
    pickle.dump(all_errors_sbm, open('all_errors_sbm_rank1_p6_alpha%i_20191904.pkl'%(int(10*alpha)), 'wb'))

print("done rank1 sbms (simulation set 1)")
