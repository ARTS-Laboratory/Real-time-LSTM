import joblib
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.random import randint
import math
import scipy
from scipy import signal
from sklearn.metrics import mean_squared_error
"""
A simple test that I wanted to do. Previously, I added a dense layer at the top
of the model. This means that for the final layer, all of the units must be 
used in a specific (linear) way to estimate pin location (a scalar). I want to
use the same model (500 us, 3 cell, 20 units), but now use two different ways
to extract a scalar. First, just take the first element of the output vector. 
That would give total freedom to the other elements of the vector to represent
other features. Another way would be to take the norm of the output vector.

TensorFlow 2.5.0
Numpy 1.19.5
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print(os.environ["CUDA_VISIBLE_DEVICES"])

#%% preprocess
#give sampling rate in sec.
def preprocess(sampling_period):
    import json
    import pickle
    import numpy as np
    import sklearn as sk
    import joblib
    f = open('data_6_with_FFT.json')
    data = json.load(f)
    f.close()
    
    acc = np.array(data['acceleration_data'])
    acc_t = np.array(data['time_acceleration_data'])
    pin = np.array(data['measured_pin_location'])
    pin_t = np.array(data['measured_pin_location_tt'])
    
    # pin contains some nan values
    from math import isnan
    for i in range(len(pin)):
        if(isnan(pin[i])):
            pin[i] = pin[i-1]
    
    resample_period = sampling_period
    pin = pin[pin_t > 1.5]
    pin_t = pin_t[pin_t > 1.5] - 1.5
    acc = acc[acc_t > 1.5]
    acc_t = acc_t[acc_t > 1.5] - 1.5
    num = int((acc_t[-1] - acc_t[0])/resample_period)
    
    resample_acc, resample_t = signal.resample(acc, num, acc_t)
    resample_pin = np.interp(resample_t, pin_t, pin)
    
    # scaling data, which means that it must be unscaled to be useful
    from sklearn import preprocessing
    acc_scaler = sk.preprocessing.StandardScaler()
    acc_scaler.fit(resample_acc.reshape(-1, 1))
    acc = acc_scaler.fit_transform(resample_acc.reshape(-1, 1)).flatten()
    pin_scaler = sk.preprocessing.StandardScaler()
    pin_scaler.fit(resample_pin.reshape(-1,1))
    pin = pin_scaler.fit_transform(resample_pin.reshape(-1,1)).flatten().astype(np.float32)
    
    # reshape for multi-input
    ds = 16
    X = np.reshape(acc[:acc.size//ds*ds], (acc.size//ds, ds))
    t = np.reshape(resample_t[:resample_t.size//ds*ds], (resample_t.size//ds, ds)).T[0]
    y = np.reshape(pin[:pin.size//ds*ds], (pin.size//ds, ds)).T[0]
    
    X = np.expand_dims(X, 0)
    
    X_train = X[:,t<30.7]
    y_train = y[t<30.7]
    t_train = t[t<30.7]
    
    X_test = X[:,t>30.7]
    y_test = y[t>30.7]
    t_test = t[t>30.7]
    
    return (X, X_train, X_test), (y, y_train, y_test), (t, t_test, t_train), pin_scaler, acc_scaler

def split_train_random(X_train, y_train, batch_size, train_len):
    run_size = X_train.shape[1]
    indices = [randint(0, run_size - train_len) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[0,index:index+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index+train_len] for index in indices]))
    return X_mini, y_mini

# use the formula SNR= (A_signal/A_noise)_rms^2. returned in dB
def signaltonoise(signal, noisy_signal, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
#%%

# keys tuples of output period and unit structure e.g. (1000, [25,25])
# contains pred of entire forward pass. use t > 30.7 to get validation portion

output_period = 500 # in us
units = 20

sample_period = output_period*10**-6/16
(X, X_train, X_test), (y, y_train, y_test), \
    (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)

# .1 sec training every time
X_mini, y_mini = split_train_random(X_train, y_train, 10000, int(.1/(output_period*10**-6)))

#%%
print("Training dense top model...")
dense_top_model = keras.Sequential((
    keras.layers.LSTM(units, return_sequences=True, input_shape=[None, 16]),
    keras.layers.LSTM(units, return_sequences=True),
    keras.layers.LSTM(units, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))
dense_top_model.compile(loss="mse",
    optimizer="adam",
)
dense_top_model.fit(X_mini, y_mini, epochs=70)

pred = pin_scaler.inverse_transform(dense_top_model.predict(X)[0])
true = pin_scaler.inverse_transform(np.expand_dims(y,-1))
snr = signaltonoise(true, pred)
print("%f dB SNR for dense top model."%snr)
#%%
print("Training extract first element model...")
take_first = keras.layers.Lambda(lambda x: x[:,0])
take_first_model = keras.Sequential((
    keras.layers.LSTM(units, return_sequences=True, input_shape=[None, 16]),
    keras.layers.LSTM(units, return_sequences=True),
    keras.layers.LSTM(units, return_sequences=True),
    keras.layers.TimeDistributed(take_first)
))
take_first_model.compile(loss="mse",
    optimizer="adam",
)
take_first_model.fit(X_mini, y_mini, epochs=70)
pred = pin_scaler.inverse_transform(take_first_model.predict(X)).T
snr = signaltonoise(true, pred)
print("%f dB SNR for first element model."%snr)
#%%
print("Training norm of vector model...")
# norm can't produce negative values
y_norm = y_train - np.min(y)

norm = keras.layers.Lambda(lambda x: tf.norm(x, axis=1))

norm_vector_model = keras.Sequential((
    keras.layers.LSTM(units, return_sequences=True, input_shape=[None, 16]),
    keras.layers.LSTM(units, return_sequences=True),
    keras.layers.LSTM(units, return_sequences=True),
    keras.layers.TimeDistributed(norm)
))
norm_vector_model.compile(loss="mse",
    optimizer="adam",
)

X_norm_mini, y_norm_mini = split_train_random(X_train, y_norm, 10000, int(.1/(output_period*10**-6)))
norm_vector_model.fit(X_norm_mini, y_norm_mini, epochs=70)
pred = pin_scaler.inverse_transform(norm_vector_model.predict(X)).T
pred = pred + np.min(y)
snr = signaltonoise(true, pred)
print("%f dB SNR for norm of vector model."%snr)