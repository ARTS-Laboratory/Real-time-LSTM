import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import math
from scipy import signal
from sklearn.metrics import mean_squared_error
import time
"""
Training a model which will be put onto the PYNQ board

TensorFlow 2.5.0
Numpy 1.19.5
"""

#%% preprocess
#give sampling rate in sec.
def preprocess(resample_period):
    import json
    f = open('data_6_with_FFT.json')
    data = json.load(f)
    f.close()
    
    acc = np.array(data['acceleration_data'])
    acc_t = np.array(data['time_acceleration_data'])
    pin = np.array(data['measured_pin_location'])
    pin_t = np.array(data['measured_pin_location_tt'])
    
    # pin contains some nan values. replace with nearby non-nan values
    from math import isnan
    for i in range(len(pin)):
        if(isnan(pin[i])):
            pin[i] = pin[i-1]
    
    # remove first 1.5 sec where nothing happens
    pin = pin[pin_t > 1.5]
    pin_t = pin_t[pin_t > 1.5] - 1.5
    acc = acc[acc_t > 1.5]
    acc_t = acc_t[acc_t > 1.5] - 1.5
    num = int((acc_t[-1] - acc_t[0])/resample_period)
    
    # resample acceleration and pin to desired rate
    acc, resample_t = signal.resample(acc, num, acc_t)
    pin = np.interp(resample_t, pin_t, pin)
    
    # rescale data by dividing by standard deviation
    X_std = np.std(acc)
    y_std = np.std(pin)
    acc = acc/X_std
    pin = pin/y_std
    
    # reshape for multi-input
    ds = 16
    X = np.reshape(acc[:acc.size//ds*ds], (acc.size//ds, ds))
    #downsample pin location by 16
    y = np.reshape(pin[:pin.size//ds*ds], (pin.size//ds, ds)).T[0]
    t = np.reshape(resample_t[:resample_t.size//ds*ds], (resample_t.size//ds, ds)).T[0]
    
    y = y.reshape(1, -1, 1)
    X = np.expand_dims(X, 0)
    
    return (X, y, t, X_std, y_std)

# signal to noise ratio
def signaltonoise(sig, pred, dB=True):
    noise = sig - pred
    a_sig = np.sqrt(np.mean(np.square(sig)))
    a_noise = np.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*np.log10(snr)
#%% preprocess dataset
output_period = 500 # in us

sample_period = output_period*10**-6/16
(X, y, t, X_std, y_std) = preprocess(sample_period)

# train individually on 400-sample subsequences of the dataset
train_len = 400

batches = X.shape[1]//train_len

X_train = X[:,:batches*train_len,:]
y_train = y[:,:batches*train_len,:]
X_train = X_train.reshape(batches, train_len, 16)
y_train = y_train.reshape(batches, train_len, 1)
#%%

# shape of the model. This has 3 sequential cells with 20 units each.
model_shape = [20, 20, 20]

# create keras model
model = keras.Sequential(keras.layers.InputLayer(input_shape=[None, 16]))
for s in model_shape:
    model.add(keras.layers.LSTM(s, return_sequences=True, stateful=False))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))


model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
)


start_time = time.perf_counter()
model.fit(
    X_train, y_train,
    batch_size=8,
    shuffle=True,
    epochs=75, 
    # validation_data = (X, y),
)
stop_time = time.perf_counter()
training_time = stop_time - start_time

y_pred = model.predict(X)

snr = signaltonoise(y, y_pred)
print("SNR: %f dB"%snr)
#%% results
plt.figure(figsize=(6, 2.5))
plt.plot(t, y.flatten()*y_std, label='reference')
plt.plot(t, y_pred.flatten()*y_std, label='prediction')
plt.xlim((0, t[-1]))
plt.ylim((.045, .2))
plt.xlabel('time (s)')
plt.ylabel('fixity location (m)')
plt.tight_layout()
plt.show()