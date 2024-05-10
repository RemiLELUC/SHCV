### Import Libraries
import numpy as np
from time import time
from sliced_wasserstein import MC_gauss, CVNN_gauss, lowerCV_gauss, upperCV_gauss, SHCV_gauss

# Generate Data
d = 3             # dimension of the problem
np.random.seed(0)
mu_X = np.random.normal(loc=1,scale=1,size=d)  # mean of start gaussian
mu_Y = np.random.normal(loc=1,scale=1,size=d) 

tabX = np.random.normal(size=(d,d))
tabY = np.random.normal(size=(d,d))
cov_X = (tabX.T)@tabX
cov_Y = (tabY.T)@tabY

# Function to integrate
def f_gauss(x):
    return (x@(mu_X-mu_Y))**2 + (np.sqrt(x@cov_X@(x.T))-np.sqrt(x@cov_Y@(x.T)))**2


if d==3:
    MAX_DEG = 17
if d==5:
    MAX_DEG = 7
if d==6:
    MAX_DEG = 5
if d==10:
    MAX_DEG = 5
if d==15:
    MAX_DEG = 5
if d==20:
    MAX_DEG = 5
    
Phi = SphericalHarmonics(dimension=d,degrees=MAX_DEG)
 
n_list = np.logspace(2+np.log10(2),3+np.log10(2),20).astype(int)  # d=15

print(n_list)

### Simples tests ###
# MC estimate
t = time()
I_mc = MC_gauss(f=f_gauss,n_dim=d,seed=2024,L=500)
t_end = time()-t
print('I_mc=',I_mc)
print('time=',t_end)

# lower-CV estimate
t = time()
I_lo = lowerCV_gauss(f=f_gauss,mu_X=mu_X, mu_Y=mu_Y,
                                n_dim=d,seed=2024,L=500)
t_end = time()-t
print('I_lo=',I_lo)
print('time=',t_end)

# upper-CV estimate
t = time()
I_up = upperCV_gauss(f=f_gauss,mu_X=mu_X, mu_Y=mu_Y,cov_X=cov_X,
                     cov_Y=cov_Y,n_dim=d,seed=2024,L=500)
t_end = time()-t
print('I_up=',I_up)
print('time=',t_end)

# SHCV estimate
t = time()
I_cv = SHCV_gauss(f=f_gauss,n_dim=d,seed=2024,L=500,Phi=Phi)
print('I_shcv=',I_cv)
t_end = time()-t
print('time=',t_end)

# CVNN estimate
t = time()
I_nn = CVNN_gauss(f=f_gauss,n_dim=d,seed=2024,L=1000,Nmc=int(1000**(1+2/d)))
print('I_nn=',I_nn)
t_end = time()-t
print('time=',t_end)

### Run on 100 independent runs
N_exp = 100
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
        I_mc = MC_gauss(f=f_gauss,n_dim=d,seed=s,L=n)
        t_end = time()-t0
        times_mc[s] = t_end
        res_mc[s] = I_mc
        
        t0 = time()
        I_cv = SHCV_gauss(f=f_gauss,n_dim=d,seed=s,L=n,Phi=Phi)
        t_end = time()-t0
        times_cv[s] = t_end
        res_cv[s] = I_cv
        
        t0 = time()
        I_nn = CVNN_gauss(f=f_gauss,n_dim=d,seed=s,L=n,Nmc=int(n**(1+2/d)))
        t_end = time()-t0
        times_nn[s] = t_end
        res_nn[s] = I_nn
        
        t0 = time()
        I_cv_lo = lowerCV_gauss(f=f_gauss,mu_X=mu_X, mu_Y=mu_Y,
                                n_dim=d,seed=s,L=n)
        t_end = time()-t0
        times_lo[s] = t_end
        res_lo[s] = I_cv_lo
        
        t0 = time()
        I_cv_up = upperCV_gauss(f=f_gauss,mu_X=mu_X, mu_Y=mu_Y,cov_X=cov_X,
                                cov_Y=cov_Y,n_dim=d,seed=s,L=n)
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
