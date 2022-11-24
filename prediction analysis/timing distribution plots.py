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
#%% mean and std for all datasets
for dist in dists:
    d = np.repeat(np.arange(0, 150), dist)
    mean = np.mean(d)
    std = np.std(d)
    print('mean: ' + str(mean))
    print('std: ' + str(std))
#%% normalize all the datasets
for i in range(len(dists)):
    dists[i] = dists[i]/dists[i].size
#%% all distributions with linear and log y axes
dist = dists[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
ax1.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlim((0, 25))
ax1.set_xlabel(u'computation time (\u03bcs)')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlim((0, 25))
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')
plt.tight_layout()
plt.savefig('./plots/gpos multiply.png', dpi=500)

#%%
dist = dists[1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
ax1.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlim((0, 50))
ax1.set_xlabel(u'computation time (\u03bcs)')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlim((0, 50))
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')
plt.tight_layout()
plt.savefig('./plots/gpos sigmoid.png', dpi=500)
#%%
dist = dists[2]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
ax1.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlim((0, 100))
ax1.set_xlabel(u'computation time (\u03bcs)')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlim((0, 100))
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')
plt.tight_layout()
plt.savefig('./plots/rtos multiply.png', dpi=500)
#%%
dist = dists[3]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
ax1.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlim((0, 90))
ax1.set_xlabel(u'computation time (\u03bcs)')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlim((0, 90))
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')
plt.tight_layout()
plt.savefig('./plots/rtos sigmoid.png', dpi=500)
#%%
dist = dists[4]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
ax1.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlim((0, 50))
ax1.set_xlabel(u'computation time (\u03bcs)')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlim((0, 50))
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')
plt.tight_layout()
plt.savefig('./plots/priority multiply.png', dpi=500)
#%%
dist = dists[5]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
ax1.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_xlim((0, 40))
ax1.set_xlabel(u'computation time (\u03bcs)')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dist, width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_xlim((0, 40))
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')
plt.tight_layout()
plt.savefig('./plots/priority sigmoid.png', dpi=500)
#%% multiplots comparing gpos, rtos results for multiply and sigmoid
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True, sharey=True)
plt.xlim((0, 100))

ax1.bar(np.arange(0, 150), dists[0], width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
plt.yscale('log')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dists[2], width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')

plt.tight_layout()
plt.savefig('./plots/gpos_rtos multiply.png', dpi=500)
#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True, sharey=True)
plt.xlim((0, 100))

ax1.bar(np.arange(0, 150), dists[1], width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
plt.yscale('log')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dists[3], width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')

plt.tight_layout()
plt.savefig('./plots/gpos_rtos sigmoid.png', dpi=500)
#%% same thing comparing rtos and priority
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True, sharey=True)
plt.xlim((0, 100))

ax1.bar(np.arange(0, 150), dists[2], width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
plt.yscale('log')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dists[4], width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')

plt.tight_layout()
plt.savefig('./plots/rtos_priority multiply.png', dpi=500)
#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True, sharey=True)
plt.xlim((0, 100))

ax1.bar(np.arange(0, 150), dists[3], width=1, align='center', edgecolor='k', linewidth=0)
ax1.grid(True)
ax1.set_axisbelow(True)
plt.yscale('log')
ax1.set_ylabel('percentage of total instances')

ax2.bar(np.arange(0, 150), dists[5], width=1, align='center', edgecolor='k', linewidth=0)
ax2.grid(True)
ax2.set_axisbelow(True)
plt.yscale('log')
ax2.set_xlabel(u'computation time (\u03bcs)')
ax2.set_ylabel('percentage of total instances')

plt.tight_layout()
plt.savefig('./plots/rtos_priority sigmoid.png', dpi=500)
#%% 2x2 plot with gpos and rtos distributions
fig, axes = plt.subplots(2, 2, figsize=(5,4), sharex=True, sharey=True)
plt.xlim((0, 100))

axes[0,0].bar(np.arange(0, 150), dists[0], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,0].set_axisbelow(True)
plt.yscale('log')
axes[0,0].grid(True)
axes[0,0].set_ylabel('percentage of total instances')

axes[1,0].bar(np.arange(0, 150), dists[1], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,0].set_axisbelow(True)
plt.yscale('log')
axes[1,0].grid(True)
axes[1,0].set_ylabel('percentage of total instances')
axes[1,0].set_xlabel(u'computation time (\u03bcs)')

axes[0,1].bar(np.arange(0, 150), dists[2], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,1].set_axisbelow(True)
plt.yscale('log')
axes[0,1].grid(True)

axes[1,1].bar(np.arange(0, 150), dists[3], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,1].set_axisbelow(True)
plt.yscale('log')
axes[1,1].grid(True)
axes[1,1].set_xlabel(u'computation time (\u03bcs)')

plt.tight_layout()
plt.savefig('./plots/gpos_rtos2x2.png', dpi=500)
#%% 2x2 plot with rtos and priority distributions
fig, axes = plt.subplots(2, 2, figsize=(5,4), sharex=True, sharey=True)
plt.xlim((0, 100))

axes[0,0].bar(np.arange(0, 150), dists[2], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,0].set_axisbelow(True)
plt.yscale('log')
axes[0,0].grid(True)
axes[0,0].set_ylabel('percentage of total instances')

axes[1,0].bar(np.arange(0, 150), dists[3], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,0].set_axisbelow(True)
plt.yscale('log')
axes[1,0].grid(True)
axes[1,0].set_ylabel('percentage of total instances')
axes[1,0].set_xlabel(u'computation time (\u03bcs)')

axes[0,1].bar(np.arange(0, 150), dists[4], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,1].set_axisbelow(True)
plt.yscale('log')
axes[0,1].grid(True)

axes[1,1].bar(np.arange(0, 150), dists[5], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,1].set_axisbelow(True)
plt.yscale('log')
axes[1,1].grid(True)
axes[1,1].set_xlabel(u'computation time (\u03bcs)')

plt.tight_layout()
plt.savefig('./plots/rtos_priority2x2.png', dpi=500)
#%% all plots together
fig, axes = plt.subplots(2, 3, figsize=(6.5,4), sharex=True, sharey=True)
plt.xlim((0, 100))

axes[0,0].bar(np.arange(0, 150), dists[0], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,0].set_axisbelow(True)
plt.yscale('log')
axes[0,0].grid(True)
axes[0,0].set_ylabel('percentage of total instances')

axes[1,0].bar(np.arange(0, 150), dists[1], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,0].set_axisbelow(True)
plt.yscale('log')
axes[1,0].grid(True)
axes[1,0].set_ylabel('percentage of total instances')
axes[1,0].set_xlabel(u'computation time (\u03bcs)')

axes[0,1].bar(np.arange(0, 150), dists[2], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,1].set_axisbelow(True)
plt.yscale('log')
axes[0,1].grid(True)

axes[1,1].bar(np.arange(0, 150), dists[3], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,1].set_axisbelow(True)
plt.yscale('log')
axes[1,1].grid(True)
axes[1,1].set_xlabel(u'computation time (\u03bcs)')

axes[0,2].bar(np.arange(0, 150), dists[4], width=1, align='center', edgecolor='k', linewidth=0)
axes[0,2].set_axisbelow(True)
plt.yscale('log')
axes[0,2].grid(True)

axes[1,2].bar(np.arange(0, 150), dists[5], width=1, align='center', edgecolor='k', linewidth=0)
axes[1,2].set_axisbelow(True)
plt.yscale('log')
axes[1,2].grid(True)
axes[1,2].set_xlabel(u'computation time (\u03bcs)')

plt.tight_layout()
plt.savefig('./plots/all_dists2x3.png', dpi=500)