import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
import json
from math import isnan
"""
Making a simple linear model on first mode frequency
"""
def signaltonoise(signal, noisy_signal, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)

f = open('data_6_with_FFT.json')
data = json.load(f)
f.close()

acc = np.array(data['acceleration_data'])
acc_t = np.array(data['time_acceleration_data'])
pin = np.array(data['measured_pin_location'])
pin_t = np.array(data['measured_pin_location_tt'])

pin = pin[pin_t > 1.5]
pin_t = pin_t[pin_t > 1.5] - 1.5
acc = acc[acc_t > 1.5]
acc_t = acc_t[acc_t > 1.5] - 1.5

fs = 1600
resample_period = 1/fs
num = int((acc_t[-1] - acc_t[0])/resample_period)
acc, acc_t = signal.resample(acc, num, acc_t)

for i in range(len(pin)):
    if(isnan(pin[i])):
        pin[i] = pin[i-1]

nperseg=256
f, t, Sxx = signal.spectrogram(acc, fs=fs, window=('tukey', 1), nperseg=nperseg, noverlap=nperseg-1)

# cut off high frequency components
max_index = np.argmax(f>100)
f = f[:max_index]
Sxx = Sxx[:max_index,:]

# plt.figure(figsize = (6.5, 3))
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('frequency (Hz)')
# plt.xlabel('time (s)')
# plt.ylim([0,120])
# plt.tight_layout()
# plt.show()

fSxx = np.repeat(f.reshape(-1, 1), t.size, axis=-1)*Sxx

num = np.trapz(fSxx, x=f, axis=0)
denom = np.trapz(Sxx, x=f, axis=0)

y = (num/denom).reshape(-1, 1)

lin_model = LinearRegression()
x = np.interp(t, pin_t, pin).reshape(-1, 1)
lin_model.fit(y, x)

x_pred = lin_model.predict(y)

plt.figure(figsize = (6.5, 3))
plt.plot(t, x, label='reference')
plt.plot(t, x_pred, label='freq. model')
plt.ylim((0.045, 0.19))
plt.ylabel('pin location (m)')
plt.xlabel('time (s)')
plt.legend(loc=1)
plt.tight_layout()
plt.savefig('./plots/frequency_pred.png', dpi=500)

snr = signaltonoise(x, x_pred)
print('SNR: ' + str(snr))