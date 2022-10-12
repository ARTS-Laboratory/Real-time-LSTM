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
Online training didn't work. Just train on whole dataset and enable early
stopping

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
#%%

# keys tuples of output period and unit structure e.g. (1000, [25,25])
# contains pred of entire forward pass. use t > 30.7 to get validation portion
dict_results = {}
output_period_scheme = [1000, 750, 500, 400, 300, 200, 100] # in us
units_scheme = [40, 35, 30, 25, 20, 18, 15, 12, 10, 8]
cells_scheme = [1, 2, 3]

# indexing each training scheme for easier starting/stopping
index = 138;
while(index < 210):
    i = index
    units_index = i % 10
    i //= 10
    output_period_index = i % 7
    i //= 7
    cells_index = i
    
    cells = cells_scheme[cells_index]
    output_period = output_period_scheme[output_period_index]
    units = units_scheme[units_index]
    
    sample_period = output_period*10**-6/16
    # calling preprocess is only necessary when output period changes, but probably not the biggest time drag
    (X, X_train, X_test), (y, y_train, y_test), \
        (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
    
    print("Now training model #%d/210, with %d us period, %d cells, %d units"
          %(index, output_period, cells, units))
    if(output_period != 100):
        # .1 sec training every time
        X_mini, y_mini = split_train_random(X_train, y_train, 10000, int(.1/(output_period*10**-6)))
        
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
        y_test = np.expand_dims(y_test, -1)
        
        model.fit(
            X_mini, y_mini,
            shuffle=False,
            epochs=1000, 
            validation_data = (X, y.reshape(1, -1, 1)),
            callbacks = [early_stopping, StateResetter()],
        )
        pred = pin_scaler.inverse_transform(model.predict(X)[0]).T
        dict_results[output_period, cells, units] = pred
        string_name = "%dus%dcells%dunits"%(output_period, cells, units)
        np.save("./prediction results/" + string_name, pred)
        model.save("./model_saves/" + string_name)
    
    index += 1
    
# for cells in cells_scheme[1:]:    
#     for output_period in output_period_scheme:
#         sample_period = output_period*10**-6/16
#         (X, X_train, X_test), (y, y_train, y_test), \
#             (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
#         for units in units_scheme:            
#             c = cells_scheme.index(cells)
#             o = output_period_scheme.index(output_period)
#             u = units_scheme.index(units)
#             model_number = u + 10*o + 70*c + 1
            
#             print("Now training model #%d/210, with %d us period, %d cells, %d units"
#                   %(model_number, output_period, cells, units))
            
            
            
#             # .1 sec training every time
#             X_mini, y_mini = split_train_random(X_train, y_train, 10000, int(.1/(output_period*10**-6)))
            
#             model = keras.Sequential(
#                 [keras.layers.LSTM(units,return_sequences=True,input_shape=[None, 16])] + 
#                 [keras.layers.LSTM(units, return_sequences = True) for i in range(cells - 1)] +
#                 [keras.layers.TimeDistributed(keras.layers.Dense(1))]
#             )
#             model.compile(loss="mse",
#                 optimizer="adam",
#                 metrics = ['accuracy']
#             )
#             y_test = np.expand_dims(y_test, -1)
            
#             model.fit(X_mini, y_mini, epochs=30)
#             pred = pin_scaler.inverse_transform(model.predict(X)[0]).T
#             dict_results[output_period, cells, units] = pred
#             string_name = "%dus%dcells%dunits"%(output_period, cells, units)
#             np.save("./prediction results/" + string_name, pred)
#             model.save("./model_saves/" + string_name)

#%% load prediction results
dict_results = {}
for cells in cells_scheme:    
    for output_period in output_period_scheme:
        for units in units_scheme:
            string_name = "%dus%dcells%dunits"%(output_period, cells, units)
            pred = np.load("./prediction results/" + string_name+".npy")
            dict_results[output_period, cells, units] = pred

#%% SNR table

#snr calculated across the entire dataset
snr_table = np.zeros((len(cells_scheme), len(output_period_scheme), len(units_scheme)))

for c in range(len(cells_scheme)):
    cells = cells_scheme[c]
    for i in range(len(output_period_scheme)):
        output_period = output_period_scheme[i]
        sample_period = output_period*(10**-6)/16
        (X, X_train, X_test), (y, y_train, y_test), \
            (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
        for j in range(len(units_scheme)):
            units = units_scheme[j]
            pred = dict_results[output_period, cells, units].T
            true = pin_scaler.inverse_transform(np.expand_dims(y,-1))
            snr = signaltonoise(true, pred)
            snr_table[c,i,j] = snr

print(snr_table)
np.savetxt("./prediction results/SNR table 1 cell.csv", snr_table[0], delimiter=',')
np.savetxt("./prediction results/SNR table 2 cell.csv", snr_table[1], delimiter=',')
np.savetxt("./prediction results/SNR table 3 cell.csv", snr_table[2], delimiter=',')
#%%
rmse_table = np.zeros((len(cells_scheme), len(output_period_scheme), len(units_scheme)))
for c in range(len(cells_scheme)):
    cells = cells_scheme[c]
    for i in range(len(output_period_scheme)):
        output_period = output_period_scheme[i]
        sample_period = output_period*(10**-6)/16
        (X, X_train, X_test), (y, y_train, y_test), \
            (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
        for j in range(len(units_scheme)):
            units = units_scheme[j]
            pred = dict_results[output_period, cells, units].T
            true = pin_scaler.inverse_transform(np.expand_dims(y,-1))
            rmse = mean_squared_error(true, pred, squared=False)
            rmse_table[c,i,j] = rmse

np.savetxt("./prediction results/RMSE table 1 cell.csv", rmse_table[0], delimiter=',')
np.savetxt("./prediction results/RMSE table 2 cell.csv", rmse_table[1], delimiter=',')
np.savetxt("./prediction results/RMSE table 3 cell.csv", rmse_table[2], delimiter=',')
#%% save model in ONNX format
# import tf2onnx
# import onnx
# import onnxruntime as rt

# sample_period = output_periods[3]*(10**-6)/16
# (X, X_train, X_test), (y, y_train, y_test), \
#     (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)

# model = keras.models.load_model("./model_saves/400us3cells15units")
# spec = (tf.TensorSpec((None, None, 16), tf.double, name="input"),)
# output_path = "./model_saves/RT-implementation.onnx"
# model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13,output_path=output_path)

# providers = ['CPUExecutionProvider']
# output_names = [n.name for n in model_proto.graph.output]
# m = rt.InferenceSession(output_path, providers=providers)
# onnx_pred = m.run(output_names, {"input": X_test})
# onnx_pred = np.array(onnx_pred)
# preds = model.predict(X_test)

# plt.figure()
# plt.plot(onnx_pred.squeeze())

# # make sure ONNX and keras have the same results
# np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-5)


# np.savetxt("preprocessed_DROPBEAR_500us_X.csv", X.squeeze(), delimiter=',');
# np.savetxt("preprocessed_DROPBEAR_500us_t.csv", t, delimiter=',')
# np.savetxt("preprocessed_DROPBEAR_500us_y.csv", y, delimiter=',')
# np.savetxt("model output.csv", y_pred, delimiter=',')
# np.savetxt("model_prediction.csv", onnx_pred.squeeze(), delimiter=',')
