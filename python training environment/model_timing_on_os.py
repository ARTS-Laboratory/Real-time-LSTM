import tensorflow.keras as keras
from time import perf_counter
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
"""
Testing the model timing on a GPOS (Windows)
"""
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
#%% load model and resave weights
# model = keras.models.load_model('./model_saves/500us3cells15units')
# model.save_weights('./model_saves/500us3cells15unitsweights')
#%%
units = 15
model = keras.Sequential([
    keras.layers.LSTM(units,return_sequences=True,input_shape=[None, 16]),
    keras.layers.LSTM(units,return_sequences=True),
    keras.layers.LSTM(units,return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
model.load_weights('./model_saves/500us3cells15unitsweights')
model.compile()
#%%
sample_period = 500*10**-6/16
(X, X_train, X_test), (y, y_train, y_test), \
    (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
#%%
print('beginning prediction...')
start_time = perf_counter()
y_pred = model.predict(X)
stop_time = perf_counter()
tot_time = stop_time - start_time
avg_time = tot_time/X.shape[1]

print('full pass time: ' + str(tot_time) + ' s')
print('per time step: ' + str(avg_time*10**6) + ' us')
#%%
plt.figure()
plt.plot(y_pred.flatten())
#%% element by element
units = 15
model = keras.Sequential([
    keras.layers.LSTM(units,return_sequences=True, batch_input_shape=[1, None, 16], stateful=True),
    keras.layers.LSTM(units,return_sequences=True, stateful=True),
    keras.layers.LSTM(units,return_sequences=True, stateful=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
model.load_weights('./model_saves/500us3cells15unitsweights')
#%%
time_steps = X.shape[1]
timing_dist = np.zeros((X.shape[1]))
print('beginning prediction...')
for i in range(time_steps):
    x = X[:,i:i+1,:]
    start_time = perf_counter()
    yp = model.predict(x)
    stop_time = perf_counter()
    timing_dist[i] = stop_time - start_time
    # print(i)
    if(i%1000 == 0):
        print('%f percent complete'%round(i/time_steps))

avg_time = np.average(timing_dist)
std = np.std(timing_dist)
full_pass_time = np.sum(timing_dist)
print('full pass time: ' + str(full_pass_time) + ' s')
print('per time step: ' + str(avg_time*10**6) + ' us')
print('standard dev.: ' + str(std*10**6) + ' us')