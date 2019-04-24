from covariates_gclass import *
from tqdm import tqdm as tqdm
import seaborn as sns
sns.set()
import _pickle as pickle
import time

np.random.seed(2)

pi0 = 0.5
seed_ratio = 0.1

d = 2
a1 = 4*np.ones(d)
a2 = 6*np.ones(d)
beta1 = [a1, a2] # beta1 = [[a1, a2, a3, .., ad], [b1, b2, .., bd]]
beta2 = [a2, a1]
beta_params = [beta1, beta2]

n = [75, 100, 150, 200, 300, 400, 500]

#- 2block SBM
print("begin simulation set 1")

M = 200
B = np.array([
        [0.7, 0.4]
        [0.4, 0.2]
])

cs = np.arange(0.2, 0.8, step=0.1)
print(cs)

n = [75, 100, 150, 200, 300, 400, 500]
for c, alpha in enumerate(tqdm(cs)):
    temp_B = B + np.diag(c*np.ones(2))
    all_errors_sbm = []

    for i in range(len(n)):
        errors_sbm = [[] for i in range(5)]
        for j in range(M):
            try:
                temp_sbm = simulation(n[i], 0.5, temp_B, beta_params, cond_ind=True, errors=errors_sbm, smooth=True)
                errors_sbm = temp_sbm
                failed = False
            except:
                print('fail, %i, %1.2f'%(k, alpha))
                failed = True
            
            if failed:
                time.sleep(5)

        all_errors_sbm.append(errors_sbm)

    plot_errors(n, all_errors_sbm, labels = ['qda', 'hhrf', 'hhknn', 'rf', 'knn'], png_title = 'DCSBM_20192404%i'%(int(10*c)))
    pickle.dump(all_errors_sbm, open('all_errors_sbm_DCSBM_20192404.pkl'%(int(10*c)), 'wb'))

print("done simulation set 1")
