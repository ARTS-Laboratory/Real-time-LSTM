import numpy as np
import matplotlib.pyplot as plt
"""
Plots for timing distributions, GPOS and RTOS (regular) and RTOS (priority),
also calculating mean and standard deviations. 
"""

plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')

#%%
gpos_mul = np.loadtxt('./timing distributions/gpos multiply.csv', delimiter=',', dtype=np.int)
gpos_sig = np.loadtxt('./timing distributions/gpos sigmoid.csv', delimiter=',', dtype=np.int)
rtos_mul = np.loadtxt('./timing distributions/rtos multiply.csv', delimiter=',', dtype=np.int)
rtos_sig = np.loadtxt('./timing distributions/rtos sigmoid.csv', delimiter=',', dtype=np.int)
prio_mul = np.loadtxt('./timing distributions/priority multiply.csv', delimiter=',', dtype=np.int)
prio_sig = np.loadtxt('./timing distributions/priority sigmoid.csv', delimiter=',', dtype=np.int)


dists = [gpos_mul, gpos_sig, rtos_mul, rtos_sig, prio_mul, prio_sig]

for i in range(len(dists)):
    dists[i] = dists[i]/dists[i].size
#%% mean and std for all datasets

#%% 


#%% 2x2 plot with gpos and rtos distributions
fig, axes = plt.subplots(2, 2, figsize=(5,5), sharex=True, sharey=True)
plt.xlim((0, 100))

axes[0,0].bar(np.arange(0, 150), dists[0], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,0].set_axisbelow(True)
plt.yscale('log')
axes[0,0].set_ylabel('percentage of total instances')

axes[1,0].bar(np.arange(0, 150), dists[1], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,0].set_axisbelow(True)
plt.yscale('log')
axes[1,0].set_ylabel('percentage of total instances')

axes[0,1].bar(np.arange(0, 150), dists[2], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,1].set_axisbelow(True)
plt.yscale('log')
axes[0,1].set_xlabel(u'forward pass time (\u03bcs)')

axes[1,1].bar(np.arange(0, 150), dists[3], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,1].set_axisbelow(True)
plt.yscale('log')
axes[1,1].set_xlabel(u'forward pass time (\u03bcs)')
plt.savefig('./plots/gpos_rtos2x2.png', dpi=500)

