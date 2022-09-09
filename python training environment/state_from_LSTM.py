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
import sklearn as sk
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
# >>> 21.931982 dB SNR for dense top model.
plt.figure(figsize=(6, 4))
plt.plot(t, pred, label="model prediction")
plt.plot(t, true, label="reference", alpha=.8)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/dense top model pred.png", dpi=500)
#%%
print("Training extract first element model...")
normal_scaler = sk.preprocessing.MinMaxScaler()
normal_scaler.fit(np.reshape(y, (-1,1)))
y_first = normal_scaler.fit_transform(y_train.reshape(-1,1)).flatten()

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

X_first_mini, y_first_mini = split_train_random(X_train, y_first, 10000, int(.1/(output_period*10**-6)))

take_first_model.fit(X_first_mini, y_first_mini, epochs=70)
pred = take_first_model.predict(X)
pred = normal_scaler.inverse_transform(pred)
pred = pin_scaler.inverse_transform(pred).T
snr = signaltonoise(true, pred)
print("%f dB SNR for first element model."%snr)
# >>> 22.492193 dB dB SNR for first element model.
plt.figure(figsize=(6, 4))
plt.plot(t, pred, label="model prediction")
plt.plot(t, true, label="reference", alpha=.8)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/first element model pred.png", dpi=500)
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
pred = norm_vector_model.predict(X)
pred = pred + np.min(y)
pred = pin_scaler.inverse_transform(pred).T
snr = signaltonoise(true, pred)
print("%f dB SNR for norm of vector model."%snr)
# >>> 24.030725 dB SNR for norm of vector model.
plt.figure(figsize=(6, 4))
plt.plot(t, pred, label="model prediction")
plt.plot(t, true, label="reference", alpha=.8)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/vector norm model pred.png", dpi=500)
#%% these classes don't work and i don't know why. i replaced them with just the 
# lambda layers then a dense layer (without activation)

# class TakeFirstElement(keras.layers.Layer):
    
#     def __init__(self):
#         super(TakeFirstElement, self).__init__()
#         self.scaler = self.add_weight(
#             shape=(1,),
#             name='scaler',
#             trainable=True
#         )
    
#     def call(self, x):
#         return self.scaler * x[:,0]
        
# class NormVector(keras.layers.Layer):
    
#     def __init__(self):
#         super(NormVector, self).__init__()
#         self.scaler = self.add_weight(
#             shape=(1,1),
#             name='scaler',
#             trainable=True
#         )
    
# #     def compute_output_shape(input_shape):
# #         return [input_shape[0], 1]
        
    
#     def call(self, x):
#         return self.scaler * tf.norm(x, axis=0)

# make the models. 2 LSTM layers with 15 units then changing the top. For the first element and norm models, a dense layer 
# used just to scale the result and add a bias (one weight, one bias)
dense_top_model = keras.Sequential((
    keras.layers.LSTM(15, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))
first_element_model = keras.Sequential((
    keras.layers.LSTM(15, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Lambda(keras.layers.Lambda(lambda x: x[:,0]))),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))
norm_vector_model = keras.Sequential((
    keras.layers.LSTM(15, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: tf.norm(x, axis=1))),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error

def generate_time_series(batch_size, n_steps, y_type = 'period'):
    T = np.random.rand(1, batch_size, 1) * 8 + 2
    phase = np.random.rand(1, batch_size, 1)*2*np.pi
    A = np.random.rand(1, batch_size, 1)*9.8 + .2
    time = np.linspace(0, n_steps, n_steps)
    series = A * np.sin((time - phase)*2*np.pi/T)
    series += 0.1 * (np.random.rand(1, batch_size, n_steps) - .5)
    rtrn = np.expand_dims(np.squeeze(series.astype(np.float32)), axis=2)
    if(y_type == 'amplitude'):
        return rtrn, A.flatten()
    if(y_type == 'frequency'):
        return rtrn, 1/T.flatten()
    if(y_type == 'next_element'):
        return rtrn[:,:,:-1], rtrn[:,:,-1]
    return rtrn, T.flatten()


dense_top_model = keras.Sequential((
    keras.layers.LSTM(15, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))
first_element_model = keras.Sequential((
    keras.layers.LSTM(15, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Lambda(keras.layers.Lambda(lambda x: x[:,0]))),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))
norm_vector_model = keras.Sequential((
    keras.layers.LSTM(15, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: tf.norm(x, axis=1))),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
))

models = [dense_top_model, first_element_model, norm_vector_model]

freq_rmse = [0,0,0] # this will fill with RMSE for each model type
freq_val = [] # append validation losses to this list
np.random.seed(42)
n_steps = 75
X, y = generate_time_series(10000, n_steps + 1, y_type='frequency')
X_train = X[:7000]; y_train = y[:7000]
X_test = X[7000:]; y_test = y[7000:]

for i in range(3):
    model = models[i]
    model.compile(
        loss="mse",
        optimizer="adam",
    )
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
    freq_val.append(hist.history['val_loss'])
    
    pred = model.predict(X_test)[:,-1].flatten()
    rmse = mean_squared_error(y_test, pred, squared=False)
    freq_rmse[i] = rmse

