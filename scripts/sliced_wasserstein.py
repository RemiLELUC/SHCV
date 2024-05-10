########################
### Import Libraries ###
########################

import ot
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neighbors import KDTree
from qmc import i4_sobol_generate, halton
from scipy.special import ndtri
import scipy.linalg as spl

###########################
### Implement functions ###
###########################


#####################
# I. Exact Gaussian #
#####################
def MC_gauss(f, n_dim, seed, L=10):
    """ Computes the Sliced-Wasserstein distance of order 2, between exact
    Gaussian distributions via integrand f, using L uniform random projections on the unit sphere
    Params:
    @f    (func): integrand f_{\mu,\nu}
    @n_dim (int): dimension of the problem
    @seed  (int): random seed for reproducibility
    @L     (int): number of random projections
    Returns:
    @sw_mc  (float): MC estimate of the SW distance
    """
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  
    eval_f = np.array([f(t) for t in theta])
    sw_mc = np.mean(eval_f)
    return sw_mc

def QMC_gauss(f, n_dim, L=10, seq='halton'):
    """ Computes the Sliced-Wasserstein distance of order 2, between exact
    Gaussian distributions via integrand f, using QMC sets on the unit sphere
    Params:
    @f    (func): integrand f_{\mu,\nu}
    @n_dim (int): dimension of the problem
    @L     (int): number of projections
    @seq   (str): 'halton' or 'sobol' (QMC points to use)
    Returns:
    @sw_qmc  (float): QMC estimate of the SW distance
    """
    if seq=='halton':
        x_qmc = halton(dim=n_dim,n_sample=L)
    elif seq=='sobol':
        x_qmc = (i4_sobol_generate(dim_num=n_dim,n=L))[1:,:]
    theta = ndtri(x_qmc)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  
    eval_f = np.array([f(t) for t in theta])
    sw_qmc = np.mean(eval_f)
    return sw_qmc
    

def RQMC_gauss(f, n_dim, seed, L=10, seq='halton'):
    """ Computes the Sliced-Wasserstein distance of order 2, between exact
    Gaussian distributions via integrand f, using RQMC sets on the unit sphere
    Params:
    @f    (func): integrand f_{\mu,\nu}
    @n_dim (int): dimension of the problem
    @L     (int): number of projections
    @seq   (str): 'halton' or 'sobol' (QMC points to use)
    Returns:
    @sw_rqmc  (float): RQMC estimate of the SW distance
    """
    if seq=='halton':
        x_qmc = halton(dim=n_dim,n_sample=L)
    elif seq=='sobol':
        x_qmc = (i4_sobol_generate(dim_num=n_dim,n=L))[1:,:]
    theta = ndtri(x_qmc)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None] 
    np.random.seed(seed)
    X_sti = np.random.normal(size=(n_dim,n_dim))
    V = X_sti@np.linalg.inv(spl.sqrtm((X_sti.T)@X_sti))
    theta = theta@V
    eval_f = np.array([f(t) for t in theta])
    sw_rqmc = np.mean(eval_f)
    return sw_rqmc
    
    
def CVNN_gauss(f, n_dim, seed, L=10, p=2, Nmc=1000):
    """ Computes the Sliced-Wasserstein distance of order 2, between exact
    Gaussian distributions via integrand f using CVNN as in (Leluc et al., 2023)
    Params:
    @f    (func): integrand f_{\mu,\nu}
    @n_dim (int): dimension of the problem
    @seed        (int): random seed for reproducibility
    @L           (int): number of random projections
    @p           (int): order of SW distance
    @Nmc         (int): nb of particles for NN approximation
    Returns:
    @mcnn  (float): CVNN estimate of the SW distance
    """
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  
    eval_sw = np.array([f(t) for t in theta])
    mc = np.mean(eval_sw)
    ## Nearest Neighbor part KDTree
    # instance of KDTree to compute k-nearest neighbors
    kdt = KDTree(theta, leaf_size = 10, metric='euclidean')
    # query the tree for the k nearest neighbors
    mask_nn = kdt.query(theta, k=2, return_distance=False)[:,1]
    # compute evaluations 𝜑̂ (X1),...,𝜑̂ (Xn)
    mchat = np.mean(eval_sw[mask_nn])
    # evaluate integral of 𝜑̂  with MC of size N=n^2
    theta_mc = np.random.default_rng(seed=seed).normal(size=(Nmc, n_dim))
    theta_mc = theta_mc / (np.sqrt((theta_mc ** 2).sum(axis=1)))[:, None]  
    mask2 = kdt.query(theta_mc, k=1, return_distance = False)
    mchatmc = np.mean(eval_sw[mask2])
    # compute CVNN estimate
    mcnn = mc - (mchat - mchatmc)
    return mcnn

def lowerCV_gauss(f, mu_X, mu_Y, n_dim, seed, L=10):
    """
    Computes the Sliced-Wasserstein distance of order 2, between exact
    Gaussian distributions via integrand f using lower-CV as in (Nguyen and Ho, 2023)
    Params:
    @f       (func): integrand f_{\mu,\nu}
    @mu_X (d-array): true mean of source Gaussian
    @mu_Y (d-array): true mean of target Gaussian
    @n_dim    (int): dimension of the problem
    @seed     (int): random seed for reproducibility
    @L        (int): number of random projections
    Returns:
    @res  (float): CV_low estimate of the SW distance
    """
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  
    eval_f = np.array([f(t) for t in theta])
    H = np.zeros((L,1))
    z = mu_X-mu_Y
    for i in range(L):
        H[i] = (theta[i]@z)**2 - np.linalg.norm(z,ord=2)**2/n_dim
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(X=H,y=eval_f)
    res = (model.intercept_)
    return res

def upperCV_gauss(f, mu_X, mu_Y, cov_X, cov_Y, n_dim, seed, L=10):
    """
    Computes the Sliced-Wasserstein distance of order 2, between exact
    Gaussian distributions via integrand f using upper-CV as in (Nguyen and Ho, 2023)
    Params:
    @f         (func): integrand f_{\mu,\nu}
    @mu_X   (d-array): true mean of source Gaussian
    @mu_Y   (d-array): true mean of target Gaussian
    @mu_X (dxd-array): true covariance of source Gaussian
    @mu_Y (dxd-array): true covariance of target Gaussian
    @n_dim      (int): dimension of the problem
    @seed       (int): random seed for reproducibility
    @L          (int): number of random projections
    Returns:
    @res  (float): CV_up estimate of the SW distance
    """
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None] 
    eval_f = np.array([f(t) for t in theta])
    H = np.zeros((L,1))
    z = (mu_X-mu_Y).reshape(-1,1)
    M = z@z.T + cov_X + cov_Y
    for i in range(L):
        H[i] = (theta[i]@M@theta[i].T) - np.trace(M)/n_dim
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(X=H,y=eval_f)
    res = (model.intercept_)
    return res

def SHCV_gauss(f, n_dim, seed, L=10, Phi=None):
    """ Computes the Sliced-Wasserstein distance of order 2, between  between exact
    Gaussian distributions via integrand f, using Spherical Harmonics 
    Control Variates based on OLSMC. This method implements the linear integration
    rule with weights that do not depend on the integrand.
    Params:
    @f    (func): integrand f_{\mu,\nu}
    @n_dim (int): dimension of the problem
    @seed  (int): random seed for reproducibility
    @L     (int): number of random projections
    @Phi  (func): func to get spherical harmonics (control variates)
    Returns:
    @res  (float): SHCV estimate of the SW distance
    """
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]
    eval_f = np.array([f(t) for t in theta])
    # Returns all spherical harmonics of even degrees
    H_full = Phi(theta)
    H = H_full[:, 1:]
    G_inv = np.linalg.inv((H.T)@H)
    Pi = H@G_inv@(H.T)
    W = np.eye(L)-Pi
    ones = np.ones(L)
    num = W@ones
    denom = (ones.T)@W@ones
    weights = num/denom
    res = weights@eval_f
    return res

#############################
# II. Sampled distributions #
#############################

def SW_MC(X, Y, seed, L=10, p=2):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using L uniform random projections on the unit sphere
    Params:
    @X  (array n x d): source samples
    @Y  (array n x d): target samples
    @seed       (int): random seed for reproducibility
    @L          (int): number of random projections
    @p          (int): order of SW distance
    Returns:
    @sw_dist  (float): MC estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None] 
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    sw_dist = np.mean(eval_sw)** (1/order)
    return sw_dist


def lower_SW(X, Y, seed, L=10, p=2):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using lower-CV as in (Nguyen and Ho, 2023)
    Params:
    @X  (array n x d): source samples
    @Y  (array n x d): target samples
    @seed       (int): random seed for reproducibility
    @L          (int): number of random projections
    @p          (int): order of SW distance
    Returns:
    @sw_upper  (float): CV_upper estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    sw_dist = np.mean(eval_sw)
    # build lower control variate
    bar_x = np.mean(X,axis=0)
    bar_y = np.mean(Y,axis=0)
    C = ((theta@bar_x) - (theta@bar_y))**2
    b = np.sum((bar_x - bar_y)**2)/n_dim
    num = np.mean(np.multiply(eval_sw-sw_dist,C-b))
    denom = np.mean((C-b)**2)
    gamma = num/denom
    sw_lower = sw_dist - gamma*np.mean(C-b)
    return sw_lower**(1/order)

def upper_SW(X, Y, seed, L=10, p=2):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using upper-CV as in (Nguyen and Ho, 2023)
    Params:
    @X  (array n x d): source samples
    @Y  (array n x d): target samples
    @seed       (int): random seed for reproducibility
    @L          (int): number of random projections
    @p          (int): order of SW distance
    Returns:
    @sw_lower  (float): CV_lower estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None] 
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    sw_dist = np.mean(eval_sw)
    # build upper control variate
    bar_x = np.mean(X,axis=0)
    bar_y = np.mean(Y,axis=0)
    C1 = ((theta@bar_x) - (theta@bar_y))**2
    C2 = np.mean((theta@(X-bar_x).T)**2,axis=1)
    C3 = np.mean((theta@(Y-bar_y).T)**2,axis=1)
    C = C1 + C2 + C3
    b1 = np.sum((bar_x - bar_y)**2)
    b2 = np.mean(np.linalg.norm((X-bar_x),axis=1)**2)
    b3 = np.mean(np.linalg.norm((Y-bar_y),axis=1)**2)
    b = (b1 + b2 + b3)/n_dim
    num = np.mean(np.multiply(eval_sw-sw_dist,C-b))
    denom = np.mean((C-b)**2)
    gamma = num/denom
    sw_upper = sw_dist - gamma*np.mean(C-b)
    return sw_upper**(1/order)


def SW_CV(X, Y, seed, L=10, p=2, Phi=None):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using Spherical Harmonics Control Variates based on OLSMC.
    If the number of control variates is very large, use LassoMC instead.
    Params:
    @X   (array n x d): source samples
    @Y   (array n x d): target samples
    @seed        (int): random seed for reproducibility
    @L           (int): number of random projections
    @p           (int): order of SW distance
    @Phi        (func): func to get spherical harmonics (control variates)
    Returns:
    @res  (float): SHCV estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None] 
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    eval_sw = ot.wasserstein_1d(u_values=xproj,v_values=yproj,p=p)
    H = Phi(theta)
    if L<=H.shape[1]:
        model=LassoCV()
    else:
        model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(X=H,y=eval_sw)
    res = (model.intercept_)
    return res**(1/order)

def SHCV(X, Y, seed, L=10, p=2, Phi=None):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using Spherical Harmonics Control Variates based on OLSMC.
    This method implements the linear integration rule with weights.
    Params:
    @X   (array n x d): source samples
    @Y   (array n x d): target samples
    @seed        (int): random seed for reproducibility
    @L           (int): number of random projections
    @p           (int): order of SW distance
    @Phi        (func): func to get spherical harmonics (control variates)
    Returns:
    @res  (float): SHCV estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    # Returns all the spherical harmonics of even degree
    H_full = Phi(theta)
    H = H_full[:, 1:]
    G_inv = np.linalg.inv((H.T)@H)
    Pi = H@G_inv@(H.T)
    W = np.eye(L)-Pi
    ones = np.ones(L)
    num = W@ones
    denom = (ones.T)@W@ones
    weights = num/denom
    res = weights@eval_sw
    return res**(1/order)

def SW_CVNN(X, Y,seed, L=10, p=2, Nmc=1000):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using CVNN as in (Leluc et al., 2023)
    Params:
    @X   (array n x d): source samples
    @Y   (array n x d): target samples
    @seed        (int): random seed for reproducibility
    @L           (int): number of random projections
    @p           (int): order of SW distance
    @Nmc         (int): nb of particles for NN approximation
    Returns:
    @mcnn  (float): CVNN estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None] 
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean((np.abs(diff) ** order),axis=0)                     
    mc = eval_sw.mean()
    ## Nearest Neighbor part KDTree
    # instance of KDTree to compute k-nearest neighbors
    kdt = KDTree(theta, leaf_size = 10, metric='euclidean')
    # query the tree for the k nearest neighbors
    mask_nn = kdt.query(theta, k=2, return_distance=False)[:,1]
    # compute evaluations 𝜑̂ (X1),...,𝜑̂ (Xn)
    mchat = np.mean(eval_sw[mask_nn])
    # evaluate integral of 𝜑̂  with MC of size N=n^2
    theta_mc = np.random.default_rng(seed=seed).normal(size=(Nmc, n_dim))
    theta_mc = theta_mc / (np.sqrt((theta_mc ** 2).sum(axis=1)))[:, None]  # Normalize
    mask2 = kdt.query(theta_mc, k=1, return_distance = False)
    mchatmc = np.mean(eval_sw[mask2])
    # compute CVNN estimate
    mcnn = mc - (mchat - mchatmc)
    return mcnn** (1/order) #,mcnn_ols** (1/order)

def SW_QMC(X, Y, L=10, p=2, seq='halton'):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using Quasi-Monte Carlo sequences on the unit sphere
    Params:
    @X  (array n x d): source samples
    @Y  (array n x d): target samples
    @L          (int): number of random projections
    @p          (int): order of SW distance
    @seq        (str): 'halton' or 'sobol' (QMC points to use)
    Returns:
    @sw_dist  (float): QMC estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    if seq=='halton':
        x_qmc = halton(dim=n_dim,n_sample=L)
    elif seq=='sobol':
        x_qmc = (i4_sobol_generate(dim_num=n_dim,n=L))[1:,:]
    theta = ndtri(x_qmc)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    sw_dist = np.mean(eval_sw)** (1/order)
    return sw_dist


def SW_RQMC(X, Y, seed=0, L=10, p=2, seq='halton'):
    """ Computes the Sliced-Wasserstein distance of order p, between empirical distributions
    stored in X and Y, using random rotations of QMC point sets the unit sphere
    Params:
    @X  (array n x d): source samples
    @Y  (array n x d): target samples
    @seed       (int): random seed for reproducibility
    @L          (int): number of random projections
    @p          (int): order of SW distance
    Returns:
    @sw_dist  (float): RQMC estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    if seq=='halton':
        x_qmc = halton(dim=n_dim,n_sample=L)
    elif seq=='sobol':
        x_qmc = (i4_sobol_generate(dim_num=n_dim,n=L))[1:,:]
    theta = ndtri(x_qmc)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  
    # Random Rotation
    np.random.seed(seed)
    X_sti = np.random.normal(size=(n_dim,n_dim))
    V = X_sti@np.linalg.inv(spl.sqrtm((X_sti.T)@X_sti))
    theta = theta@V
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    sw_dist = np.mean(eval_sw)** (1/order)
    return sw_dist


#########################
# III. SW-based Kernels #
#########################

def SW_kernel(X1, X2,  seed=2024, L=10, p=2, c=1.0):
    """ Computes the Sliced-Wasserstein kernel between two datasets
    Params:
    @X1   (array N1 x n1 x d): N1 samples, each represents a distribution (n1 x d)
    @X2   (array N2 x n2 x d): N2 samples, each represents a distribution (n2 x d)
    @seed        (int): random seed for reproducibility
    @L           (int): number of random projections
    @p           (int): order of SW distance
    @c         (float): scaling bandwith parameter
    Returns:
    @K  (array N1 x N2): Kernel matrix K(i,j) = exp(-c * SW_2^2(x_i,x_j))
    """
    num_samples_X1, num_points_X1, dim_X1 = X1.shape
    num_samples_X2, num_points_X2, dim_X2 = X2.shape
    assert dim_X1 == dim_X2, "Dimensions of point clouds must match"
    # Precompute random directions
    rng = np.random.default_rng(seed=seed)
    thetas = rng.normal(size=(L, dim_X1))
    thetas /= np.linalg.norm(thetas, axis=1)[:, None]  # Normalize
    # Compute Sliced-Wasserstein Kernel
    kernel_matrix = np.zeros((num_samples_X1, num_samples_X2))
    for i in range(num_samples_X1):
        xproj = np.sort(np.matmul(X1[i], thetas.T), axis=0)
        for j in range(num_samples_X2):
            yproj = np.sort(np.matmul(X2[j], thetas.T), axis=0)
            eval_sw = ot.wasserstein_1d(u_values=xproj,v_values=yproj,p=p)
            sw_dist = np.mean(eval_sw) ** (1/p)
            # Fill the kernel matrix with exponential of negative squared distance
            kernel_matrix[i, j] = np.exp(-c * (sw_dist**2))
    return kernel_matrix


def SHCV_kernel(X1, X2, L=10, seed=None, Phi=None):
    """ Computes the Sliced-Wasserstein kernel between two datasets
    Params:
    @X1   (array N1 x n1 x d): N1 samples, each represents a distribution (n1 x d)
    @X1   (array N2 x n2 x d): N2 samples, each represents a distribution (n2 x d)
    @seed        (int): random seed for reproducibility
    @L           (int): number of random projections
    @Phi        (func): func to get spherical harmonics (control variates)
    Returns:
    @K_mc    (array N1 x N2): matrix K(i,j) = SW_2^2(x_i,x_j) with MC
    @K_shcv  (array N1 x N2): matrix K(i,j) = SW_2^2(x_i,x_j) with SHCV
    """
    num_samples_X1, dim_X1 = len(X1), X1[0].shape[1]-1
    num_samples_X2, dim_X2 = len(X2), X2[0].shape[1]-1
    assert dim_X1 == dim_X2, "Dimensions of point clouds must match"
    # Precompute random directions
    rng = np.random.default_rng(seed=seed)
    thetas = rng.normal(size=(L, dim_X1))
    thetas /= np.linalg.norm(thetas, axis=1)[:, None]  # Normalize
    # Compute OLS weights
    H = Phi(thetas)
    #H = H_full[:, 1:]
    G_inv = np.linalg.inv((H.T)@H)
    Pi = H@G_inv@(H.T)
    W = np.eye(L)-Pi
    ones = np.ones(L)
    num = W@ones
    denom = (ones.T)@W@ones
    weights = num/denom
    # Compute Sliced-Wasserstein Kernel
    K_mc = np.zeros((num_samples_X1, num_samples_X2))
    K_shcv = np.zeros((num_samples_X1, num_samples_X2))
    if num_samples_X1==num_samples_X2:
        for i in range(num_samples_X1):
            xproj = np.sort(np.matmul(X1[i][:,:2], thetas.T), axis=0)
            for j in range(i,num_samples_X2):
                yproj = np.sort(np.matmul(X2[j][:,:2], thetas.T), axis=0)
                eval_sw = ot.wasserstein_1d(u_values=xproj,u_weights=X1[i][:,2],
                                            v_values=yproj,v_weights=X2[j][:,2],
                                            p=2)
                sw_dist = np.mean(eval_sw) 
                shcv_dist = (weights@eval_sw)
                # Fill the kernel matrix with exponential of negative squared distance
                K_mc[i, j] = sw_dist
                K_mc[j, i] = sw_dist
                K_shcv[i, j] = shcv_dist
                K_shcv[j, i] = shcv_dist
    else:
        for i in range(num_samples_X1):
            xproj = np.sort(np.matmul(X1[i][:,:2], thetas.T), axis=0)
            for j in range(num_samples_X2):
                yproj = np.sort(np.matmul(X2[j][:,:2], thetas.T), axis=0)
                eval_sw = ot.wasserstein_1d(u_values=xproj,u_weights=X1[i][:,2],
                                            v_values=yproj,v_weights=X2[j][:,2],
                                            p=2)
                sw_dist = np.mean(eval_sw) 
                shcv_dist = (weights@eval_sw)
                # Fill the kernel matrix with exponential of negative squared distance
                K_mc[i, j] = sw_dist
                K_shcv[i, j] = shcv_dist        
    return K_mc,K_shcv


def Phi_circular(samples,L):
    """ Computes Circular Harmonics on the circle in dimension d=2 (Fourier)
    Params:
    @samples (array n x d): input samples to evaluate
    @L               (int): degree of circular harmonics
    Returns:
    @H (array n x L): matrix of Control Variates
    """
    den = np.sqrt(np.pi)
    n = len(samples)
    thetas = np.arctan2(samples[:,1],samples[:,0])
    H = np.zeros((n,L))
    for j in range(1,L+1):
        if (j & 1): # odd value of j
            w = (j//2)+1
            H[:,j-1] = np.cos(w*thetas)/den
        else:
            w = j//2
            H[:,j-1] = np.sin(w*thetas)/den
    return H


def phi_circle(L):
    """ Define Function Phi to use in SHCV estimate """
    return lambda samples: Phi_circular(samples, L)


def preprocess_image(image):
    """ Tool function for preprocessing image, convert
    an image into a 2D distributions of active pixels """
    # Find active pixels
    active_pixels = np.argwhere(image > 0)
    n_t = len(active_pixels)
    # Normalize positions to [0, 1]^2
    normalized_positions = 2*(active_pixels/7) - 1
    # Extract pixel intensities
    pixel_intensities = image[active_pixels[:, 0], active_pixels[:, 1]]
    # Compute weights
    weights = pixel_intensities / np.sum(pixel_intensities)
    # Combine normalized positions and weights
    result_matrix = np.column_stack((normalized_positions, weights))
    return result_matrix


