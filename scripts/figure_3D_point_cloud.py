### Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns

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
col_medians = 'red'

# 8: plane, 32: lamp, 35: bed
# select source and target
ind_source = 8 # 8 32 35
ind_target = 32
# load true SW value
I_true = np.load('../results/3D_point_cloud_res/I_true_{}_{}.npy'.format(str(ind_source),str(ind_target)))
# load results for boxplots
val_mc = np.load('../results/3D_point_cloud_res/val_mc_N100_ind_{}_{}.npy'.format(str(ind_source),str(ind_target)))
val_lo = np.load('../results/3D_point_cloud_res/val_lo_N100_ind_{}_{}.npy'.format(str(ind_source),str(ind_target)))
val_up = np.load('../results/3D_point_cloud_res/val_up_N100_ind_{}_{}.npy'.format(str(ind_source),str(ind_target)))
val_cv = np.load('../results/3D_point_cloud_res/val_cv_N100_ind_{}_{}.npy'.format(str(ind_source),str(ind_target)))
val_nn = np.load('../results/3D_point_cloud_res/val_nn_N100_ind_{}_{}.npy'.format(str(ind_source),str(ind_target)))
#val_rqmc = np.load('../results/3D_point_cloud_res/val_rqmc_N100_ind_{}_{}.npy'.format(str(ind_source),str(ind_target)))

### Plot Boxplot
fig, ax = plt.subplots(figsize=(5,5))
# Display Boxplotstrue
#tab = np.array(range(0, le))
bp_mc = ax.boxplot(val_mc.T,widths=0.7,
                   positions=5*np.arange(4)-1.4,
                   sym='',patch_artist=True)
bp_lo = ax.boxplot(val_lo.T,widths=0.7,
                   positions=5*np.arange(4)-0.7,
                   sym='',patch_artist=True)
bp_up = ax.boxplot(val_up.T,widths=0.7,
                   positions=5*np.arange(4),
                   sym='',patch_artist=True)
bp_nn = ax.boxplot(val_nn.T,widths=0.7,
                   positions=5*np.arange(4)+0.7,
                   sym='',patch_artist=True)
bp_cv = ax.boxplot(val_cv.T,widths=0.7,
                   positions=5*np.arange(4)+1.4,
                   sym='',patch_artist=True)

for patch in bp_mc['boxes']:
    patch.set_facecolor(col_mc)
    
for patch in bp_mc['medians']:
    patch.set_alpha(1)
    patch.set_color(col_medians)
    
for patch in bp_lo['boxes']:
    patch.set_facecolor(col_lo)
    
for patch in bp_lo['medians']:
    patch.set_color(col_medians)
    
for patch in bp_up['boxes']:
    patch.set_facecolor(col_up)
    
for patch in bp_up['medians']:
    patch.set_color(col_medians)
    
for patch in bp_nn['boxes']:
    patch.set_facecolor(col_nn)
    
for patch in bp_nn['medians']:
    patch.set_color(col_medians)    
    
for patch in bp_cv['boxes']:
    patch.set_facecolor(col_cv)
    
for patch in bp_cv['medians']:
    patch.set_color(col_medians)

# Plot legends
plt.plot([], c=col_mc, label='MC')
plt.plot([], c=col_nn, label='CVNN')
plt.plot([], c=col_lo, label='CV_low')
plt.plot([], c=col_cv, label='SHCV')
plt.plot([], c=col_up, label='CV_up')

plt.xticks(ticks=5*np.arange(4),
           labels=['100','250','500','1000'],
           fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Number of Random Projections',fontsize=15)
plt.ylabel(r'Error distribution',fontsize=15)
plt.legend(fontsize=10,ncols=3)
plt.grid(alpha=0.3,linestyle='--')
plt.xlim(left=-2.5)
#plt.ylim(bottom=-0.06)

#plt.savefig('boxplot_ot_d3.pdf',
#            transparent=True,
#            bbox_inches='tight',
#            pad_inches=0)

############ ZOOM ###############
# 8-32
y1 = -0.004
y2 = 0.0064

# 8-35
#y1 = -0.004
#y2 = 0.0064
#axins = zoomed_inset_axes(ax, 2.1, loc=4,borderpad=0.7)  # zoom = 3, location = 4 (lower right)
axins = zoomed_inset_axes(ax, 1.3, loc=4,borderpad=0.7)  # zoom = 3, location = 4 (lower right)

bpz_mc = axins.boxplot(val_mc.T,widths=0.7,
                   positions=5*np.arange(4)-1.4,
                   sym='',patch_artist=True)
bpz_lo = axins.boxplot(val_lo.T,widths=0.7,
                   positions=5*np.arange(4)-0.7,
                   sym='',patch_artist=True)
bpz_up = axins.boxplot(val_up.T,widths=0.7,
                   positions=5*np.arange(4),
                   sym='',patch_artist=True)
bpz_nn = axins.boxplot(val_nn.T,widths=0.7,
                   positions=5*np.arange(4)+0.7,
                   sym='',patch_artist=True)
bpz_cv = axins.boxplot(val_cv.T,widths=0.7,
                   positions=5*np.arange(4)+1.4,
                   sym='',patch_artist=True)

for patch in bpz_mc['boxes']:
    patch.set_facecolor(col_mc)
    
for patch in bpz_mc['medians']:
    patch.set_color(col_medians)
    
for patch in bpz_lo['boxes']:
    patch.set_facecolor(col_lo)
    
for patch in bpz_lo['medians']:
    patch.set_color(col_medians)
    
for patch in bpz_up['boxes']:
    patch.set_facecolor(col_up)
    
for patch in bpz_up['medians']:
    patch.set_color(col_medians)
    
for patch in bpz_nn['boxes']:
    patch.set_facecolor(col_nn)
    
for patch in bpz_nn['medians']:
    patch.set_color(col_medians)    
    
for patch in bpz_cv['boxes']:
    patch.set_facecolor(col_cv)
    
for patch in bpz_cv['medians']:
    patch.set_color(col_medians)
    
# sub region of the original image
x1 = 13.1
x2 = 16.8
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_xticks([])
mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec="0.4",linestyle='--')
axins.axhline(y=0, linestyle=':', color='k', alpha=0.5)
axins.grid(alpha=0.2)
ax.axhline(y=0, linestyle=':', color='k', alpha=0.5)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Uncomment to save figure
#plt.savefig('box_source{}_target{}.pdf'.format(str(ind_source),str(ind_target)),
#            transparent=True,
#           bbox_inches='tight', pad_inches=0.)
plt.show()
