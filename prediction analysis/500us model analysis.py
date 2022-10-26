import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
"""
Compare the SNR of the originial model prediction and RTOS prediction, and
error increase from model to RTOS implementation
"""
def signaltonoise(signal, noisy_signal, invert=False, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    if(not invert):
        snr = (a_sig/a_noise)**2
    else:
        snr = (a_noise/a_sig)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
y_true = np.loadtxt("./prediction/500 us pin location.csv")
y_pred = np.load("./prediction/500us2cells15units.npy").flatten()
y_rtos = np.loadtxt("./prediction/rtos500us2cells15units.csv", delimiter=',')
pin_scaler = StandardScaler()
pin_scaler.fit(y_true.reshape(-1, 1))
y_rtos = pin_scaler.inverse_transform(y_rtos.reshape(-1, 1)).flatten()
dt = 500*10**-6
t = np.array([i*dt for i, _ in enumerate(y_true)])
#%%
delay = 250
plt.figure(figsize=(6.5, 2.5))
plt.plot(t, y_true, label='reference')
plt.plot(t, y_pred, label='model')
plt.plot(t[:-delay +1], y_rtos[delay:], label='RTOS')
plt.xlabel("time (s)")
plt.ylabel("roller location (m)")
plt.ylim((0.045, .19))
plt.xlim((0,t[-1]))
plt.legend(loc=2, fancybox=False, framealpha=1)
plt.grid()
plt.tight_layout()
#%%
y_rtos = y_rtos[delay:]
y_rtos = np.append(y_rtos, np.full((y_true.size-y_rtos.size), y_rtos[-1]))

snr_model = signaltonoise(y_true, y_pred)
snr_rtos = signaltonoise(y_true, y_rtos)

print("original training SNR: " + str(snr_model))
print("real time SNR: " + str(snr_rtos))
