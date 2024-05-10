### Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

### Configure font
plt.rcParams.update({"text.usetex": True,"font.family": "serif"})

# Configure colors
pal = sns.color_palette(palette='colorblind')
col_mc = pal.as_hex()[0]
col_up = pal.as_hex()[1]
col_lo = pal.as_hex()[2]
col_nn = pal.as_hex()[3]
col_cv = pal.as_hex()[4]
col_rqmc = pal.as_hex()[5]

parser = argparse.ArgumentParser()    
parser.add_argument("--d", type=int, default=3, help="data dimension")    
parser.add_argument("--gauss_type",type=str, default='exact', help="{exact} or {sampled}")    
args = parser.parse_args()
d = args.d
gauss_type = args.gauss_type

n_list = np.logspace(2,3+np.log10(2),20).astype(int)
   
# Load results
err_mc = np.load('../results/synthetic_gaussian_{}/err_mc_d{}.npy'.format(gauss_type,str(d)))
err_lo = np.load('../results/synthetic_gaussian_{}/err_lo_d{}.npy'.format(gauss_type,str(d)))
err_up = np.load('../results/synthetic_gaussian_{}/err_up_d{}.npy'.format(gauss_type,str(d)))
err_cv = np.load('../results/synthetic_gaussian_{}/err_cv_d{}.npy'.format(gauss_type,str(d)))
err_nn = np.load('../results/synthetic_gaussian_{}/err_nn_d{}.npy'.format(gauss_type,str(d)))
err_rqmc = np.load('../results/synthetic_gaussian_{}/err_rqmc_d{}.npy'.format(gauss_type,str(d)))

t_mc = np.load('../results/synthetic_gaussian_{}/t_mc_d{}.npy'.format(gauss_type,str(d)))
t_lo = np.load('../results/synthetic_gaussian_{}/t_lo_d{}.npy'.format(gauss_type,str(d)))
t_up = np.load('../results/synthetic_gaussian_{}/t_up_d{}.npy'.format(gauss_type,str(d)))
t_cv = np.load('../results/synthetic_gaussian_{}/t_cv_d{}.npy'.format(gauss_type,str(d)))
t_nn = np.load('../results/synthetic_gaussian_{}/t_nn_d{}.npy'.format(gauss_type,str(d)))
t_rqmc = np.load('../results/synthetic_gaussian_{}/t_rqmc_d{}.npy'.format(gauss_type,str(d)))

### Plot Figure with respect to Nb of projections
plt.figure(figsize=(4,4)) # (4,4) to savefig
plt.plot(n_list,err_mc,marker='o',ms=4,color=col_mc,label='MC')
plt.plot(n_list,err_nn,marker='d',ms=4,color=col_nn,label='CVNN')
plt.plot(n_list,err_up,marker='^',ms=4,color=col_up,label='CV_up')
plt.plot(n_list,err_lo,marker='v',ms=4,color=col_lo,label='CV_low')
plt.plot(n_list,err_rqmc,marker='p',ms=4,color=col_rqmc,label='RQMC')
plt.plot(n_list,err_cv,marker='s',ms=4,color=col_cv,label='SHCV')
plt.yscale('log')
plt.xscale('log')
plt.ylim(top=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Number of projections",fontsize=14)
plt.ylabel("Root Mean Squared Error",fontsize=14)
plt.legend(ncol=3,fontsize=9)
plt.grid(alpha=0.2)
#plt.savefig('synthetic_d{}_{}.pdf'.format(str(d),gauss_type),transparent=True,
#           bbox_inches='tight', pad_inches=0)
plt.show()


### Plot Figure with respect to Computing time
plt.figure(figsize=(4,4))
plt.plot(t_mc,err_mc,marker='o',ms=4,color=col_mc,label='MC')
plt.plot(t_nn,err_nn,marker='d',ms=4,color=col_nn,label='CVNN')
plt.plot(t_up,err_up,marker='^',ms=4,color=col_up,label='CV_up')
plt.plot(t_lo,err_lo,marker='v',ms=4,color=col_lo,label='CV_low')
plt.plot(t_rqmc,err_rqmc,marker='p',ms=4,color=col_rqmc,label='RQMC')
plt.plot(t_cv,err_cv,marker='s',ms=4,color=col_cv,label='SHCV')
plt.yscale('log')
plt.xscale('log')
plt.ylim(top=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Computing time (s)",fontsize=14)
plt.ylabel("Root Mean Squared Error",fontsize=14)
plt.legend(ncol=3,fontsize=9)
plt.grid(alpha=0.2)
#plt.savefig('synthetic_d{}_{}_time.pdf'.format(str(d),gauss_type),transparent=True,
#           bbox_inches='tight', pad_inches=0)
plt.show()
