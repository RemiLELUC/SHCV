### Import libraries
import numpy as np
from time import time
from spherical_harmonics import SphericalHarmonics
from sliced_wasserstein import SW_MC, lower_SW, upper_SW, SW_CV, SW_CVNN

### Load data
A = np.load('data_3D_point_cloud.npy')

### Select shapes 8-32-35
ind_source = 8 #  32
ind_target = 32 # 35
X_sample = A[ind_source]
Y_sample = A[ind_target]

# MC estimate for I_true
t = time()
I_true = SW_MC(X=X_sample,Y=Y_sample,seed=0,L=int(1e8),p=2)
t_end = time()-t
print('I_true=',I_true)
print('time=',t_end)

# Define spherical harmonics
Phi = SphericalHarmonics(3,5) 

# Number of independent runs
N_exp = 100

# Different values of 'n=nb of random projections'
#n_list = np.logspace(1,3,20).astype(int) # for graphs
n_list = [100, 250, 500, 1000] # for boxplots
print(n_list)

# Initialize results
res_mc_total = np.zeros((len(n_list),N_exp))
res_cv_total = np.zeros((len(n_list),N_exp))
res_cv_lower = np.zeros((len(n_list),N_exp))
res_cv_upper = np.zeros((len(n_list),N_exp))
res_nn_total = np.zeros((len(n_list),N_exp))
# Initialize computing times
times_mc_total = np.zeros((len(n_list),N_exp))
times_cv_total = np.zeros((len(n_list),N_exp))
times_cv_lower = np.zeros((len(n_list),N_exp))
times_cv_upper = np.zeros((len(n_list),N_exp))
times_nn_total = np.zeros((len(n_list),N_exp))
for i,n in enumerate(n_list):
    print('n=',n)
    res_mc = np.zeros(N_exp)
    res_cv = np.zeros(N_exp)
    res_lo = np.zeros(N_exp)
    res_up = np.zeros(N_exp)
    res_nn = np.zeros(N_exp)
    times_mc = np.zeros(N_exp)
    times_cv = np.zeros(N_exp)
    times_lo = np.zeros(N_exp)
    times_up = np.zeros(N_exp)
    times_nn = np.zeros(N_exp)
    for s in range(N_exp):
        t0 = time()
        I_mc = SW_MC(X=X_sample,Y=Y_sample,seed=s,L=n,p=2)
        t_end = time()-t0
        times_mc[s] = t_end
        res_mc[s] = I_mc
        
        t0 = time()
        I_cv = SW_CV(X=X_sample,Y=Y_sample,seed=s,L=n,p=2,Phi=Phi)
        t_end = time()-t0
        times_cv[s] = t_end
        res_cv[s] = I_cv
        
        t0 = time()
        I_nn = SW_CVNN(X=X_sample,Y=Y_sample,seed=s,L=n,p=2,Nmc=int(n**(1+2/3)))
        t_end = time()-t0
        times_nn[s] = t_end
        res_nn[s] = I_nn
        
        t0 = time()
        I_cv_lo = lower_SW(X=X_sample,Y=Y_sample,seed=s,L=n,p=2)
        t_end = time()-t0
        times_lo[s] = t_end
        res_lo[s] = I_cv_lo
        
        t0 = time()
        I_cv_up = upper_SW(X=X_sample,Y=Y_sample,seed=s,L=n,p=2)
        t_end = time()-t0
        times_up[s] = t_end
        res_up[s] = I_cv_up
        
    res_mc_total[i] = res_mc
    res_nn_total[i] = res_nn
    res_cv_lower[i] = res_lo
    res_cv_upper[i] = res_up
    res_cv_total[i] = res_cv
    
    times_mc_total[i] = times_mc
    times_nn_total[i] = times_nn
    times_cv_lower[i] = times_lo
    times_cv_upper[i] = times_up
    times_cv_total[i] = times_cv
    
# Save results
#np.save('res_mc_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),res_mc_total)
#np.save('res_lo_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),res_cv_lower)
#np.save('res_up_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),res_cv_upper)
#np.save('res_nn_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),res_nn_total)
#np.save('res_cv_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),res_cv_total)

# Save computing times
#np.save('times_mc_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),times_mc_total)
#np.save('times_lo_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),times_cv_lower)
#np.save('times_up_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),times_cv_upper)
#np.save('times_nn_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),times_nn_total)
#np.save('times_cv_source{}_target{}.npy'.format(str(ind_source),str(ind_target)),times_cv_total)

val_mc = (res_mc_total-I_true)
val_lo = (res_cv_lower-I_true)
val_up = (res_cv_upper-I_true)
val_cv = (res_cv_total-I_true)
val_nn = (res_nn_total-I_true)

# Save errors for boxplots
#np.save('val_mc_N100_ind_{}_{}'.format(str(ind_source),str(ind_target)),val_mc)
#np.save('val_lo_N100_ind_{}_{}'.format(str(ind_source),str(ind_target)),val_lo)
#np.save('val_up_N100_ind_{}_{}'.format(str(ind_source),str(ind_target)),val_up)
#np.save('val_cv_N100_ind_{}_{}'.format(str(ind_source),str(ind_target)),val_cv)
#np.save('val_nn_N100_ind_{}_{}'.format(str(ind_source),str(ind_target)),val_nn)
