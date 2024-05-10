### Import Libraries

from time import time
from sklearn.datasets import load_digits
from sliced_wasserstein import preprocess_image
from sklearn.model_selection import train_test_split
from sw_estimates import phi_circle

### Load data
X,y =load_digits(return_X_y=True)
X = X.reshape(-1,8,8)
print('X shape: ',X.shape)
print('y shape: ',y.shape)

### Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=2024)
print('X_train shape:',X_train.shape)
print('X_test shape:',X_test.shape)
print('y_train shape:',y_train.shape)
print('y_test shape:',y_test.shape)

X_pt_train = []
for i in range(len(X_train)):
    ti = preprocess_image(X_train[i])
    X_pt_train.append(ti)
    
X_pt_test = []
for i in range(len(X_test)):
    ti = preprocess_image(X_test[i])
    X_pt_test.append(ti)
    
# Loop over different parameter L (nb of random projections)
for nb,L in enumerate([25,50,75,100]):
    print('*'*30)
    print('L=',L)
    Phi = phi_circle(L=n_har[nb])
    for seed in range(100):
        print('seed=',seed)
        t = time()
        SW_xx,SHCV_xx = SHCV_kernel(X1=X_pt_train, X2=X_pt_train, num_proj=L,seed=seed,Phi=Phi)
        t_end = time()
        print('Time K_xx=',t_end-t)
        np.save('SWMC_xx_split{}_L{}_s{}.npy'.format(str(split),str(L),str(seed)),SW_xx)
        np.save('SHCV_xx_split{}_L{}_s{}.npy'.format(str(split),str(L),str(seed)),SHCV_xx)
        t = time()
        SW_xy,SHCV_xy = SHCV_kernel(X1=X_pt_test, X2=X_pt_train, num_proj=L,seed=seed,Phi=Phi)
        t_end = time()
        print('Time K_xy=',t_end-t)
        np.save('SWMC_xy_split{}_L{}_s{}.npy'.format(str(split),str(L),str(seed)),SW_xy)
        np.save('SHCV_xy_split{}_L{}_s{}.npy'.format(str(split),str(L),str(seed)),SHCV_xy)