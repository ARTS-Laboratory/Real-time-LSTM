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
import time
"""
Training a model which will be put onto the PYNQ board

TensorFlow 2.5.0
Numpy 1.19.5
onnxruntime 1.11.0 (for compatibility with Numpy)
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
    y = np.reshape(pin[:pin.size//ds*ds], (pin.size//ds, ds)).T[0]
    
    y = y.reshape(1, -1, 1)
    
    X = np.expand_dims(X, 0)
    
    return (X, y, pin_scaler, acc_scaler)

"""
Training generator for online training. list all inputs. last arg should be y
"""
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, *args, train_len=400):
        self.args = args
        self.train_len = train_len
        self.length = args[0].shape[1]//train_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # F_batch = self.F[:,index*self.train_len:(index+1)*self.train_len,:]
        # lstm_batch = self.lstm_input[:,index*self.train_len:(index+1)*self.train_len,:]
        # return lstm_batch, F_batch
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        
        return rtrn[:-1], rtrn[-1] 

class StateResetter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        for layer in self.model.layers:
            if(layer.stateful):
                layer.reset_states()

class DelayEarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss',
             min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, 
             restore_best_weights=False, start_epoch = 100): # add argument for starting epoch
        super(DelayEarlyStopping, self).__init__(monitor=monitor, 
                                                 min_delta=min_delta, 
                                                 patience=patience,
                                                 verbose=0,
                                                 mode='auto',
                                                 baseline=None)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
"""
Grabs randomly from training data batch_size amount of times, returns an 
nparray with shape [batch_size, train_len, features]
"""
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
#%% preprocess dataset
output_period = 500 # in us

sample_period = output_period*10**-6/16
# calling preprocess is only necessary when output period changes, but probably not the biggest time drag
(X, y, pin_scaler, acc_scaler) = preprocess(sample_period)

# training_generator = TrainingGenerator(X, y, train_len=int(.1/(output_period*10**-6)))
training_generator = TrainingGenerator(X, y, train_len=400)
#%%
units = 40
cells = 1
string_name = "%dus%dcells%dunits"%(output_period, cells, units)

early_stopping = DelayEarlyStopping(
    start_epoch=30,
    monitor="val_loss",
    # min_delta=0.001,
    patience=10,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)

model = keras.Sequential(
    [keras.layers.LSTM(units,return_sequences=True,input_shape=[None, 16])] + 
    [keras.layers.LSTM(units, return_sequences = True) for i in range(cells - 1)] +
    [keras.layers.TimeDistributed(keras.layers.Dense(1))]
)
model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
)


start_time = time.perf_counter()
model.fit(
    training_generator,
    shuffle=True,
    epochs=1000, 
    validation_data = (X, y.reshape(1, -1, 1)),
    callbacks = [early_stopping, StateResetter()],
)
stop_time = time.perf_counter()
training_time = stop_time - start_time

pred = pin_scaler.inverse_transform(model.predict(X)[0]).T

np.save("./prediction results/" + string_name, pred)
model.save("./model_saves/" + string_name)

#%% results
true = pin_scaler.inverse_transform(np.expand_dims(y,-1))
snr = signaltonoise(true, pred)
rmse = mean_squared_error(true, pred, squared=False)
