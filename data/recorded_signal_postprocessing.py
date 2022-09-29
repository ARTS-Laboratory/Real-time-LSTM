import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
"""
post processing, validating experiment, and calculating SNR from this test
"""
data = loadtxt("recorded_signal_400us16samples.csv", delimiter=',')
prepro = loadtxt("preprocessed_DROPBEAR_X.csv", delimiter=',')
acc = loadtxt("X_test.csv", delimiter=',')
t = loadtxt("t_test.csv", delimiter=',')
#%%
acc = np.reshape(acc.T, acc.size)
t = np.reshape(t.T, t.size)
data = np.reshape(data, data.size)
dt = 400*10**-6/16
data_t = np.array([dt*i for i in range(data.size)])
#%%
dist = 5.08858
t = t - 1.5
data_t = data_t - dist
plt.figure()
plt.plot(t, acc)
plt.plot(data_t, data)
plt.xlim((0, .1))
#%%
mask = np.logical_and(data_t>0, data_t<t[-1])
data_t = data_t[mask]
data = data[mask]
data = np.reshape(data[:(data.size//16*16)], (data.size//16, 16))
data_t =  np.reshape(data_t[:(data.size//16*16)], (data.size//16, 16))

np.savetxt("noisy_DROPBEAR_X.csv", data, delimiter=',')