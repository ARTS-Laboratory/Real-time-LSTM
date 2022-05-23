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
Basically doing the same thing as train_full_model_v3 but now with a lot more
models (72). For all models we'll be leaving the impulses profile for 
validation. Instead, we are changing the sampling rate.pip 

TensorFlow 2.5.0
Numpy 1.19.5
onnxruntime 1.11.0 (for compatibility with Numpy)
"""
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
dict_results = {}
output_periods = [1000, 750, 500, 400, 300, 200, 100] # in us
unit_structures = (
    [40],[35],[20],[25,25],[20,20],[10,10],[18,18,18],[15,15,15],[8,8,8]
)

for output_period in output_periods[4:]:
    sample_period = output_period*10**-6/16
    (X, X_train, X_test), (y, y_train, y_test), \
        (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
    for units in unit_structures:
        print("Now training model %d us output, %d cells, %d units"
              %(output_period, len(units), units[0]))
        
        # .1 sec training every time
        X_mini, y_mini = split_train_random(X_train, y_train, 10000, int(.1/(output_period*10**-6)))

        model = keras.Sequential(
            [keras.layers.LSTM(units[0],return_sequences=True,input_shape=[None, 16])] + 
            [keras.layers.LSTM(i, return_sequences = True) for i in units[1:]] +
            [keras.layers.TimeDistributed(keras.layers.Dense(1))]
        )
        model.compile(loss="mse",
            optimizer="adam",
            metrics = ['accuracy']
        )
        y_test = np.expand_dims(y_test, -1)
        
        model.fit(X_mini, y_mini, epochs=20)
        pred = pin_scaler.inverse_transform(model.predict(X)[0]).T
        dict_results[output_period, len(units), units[0]] = pred
        string_name = "%dus%dcells%dunits"%(output_period, len(units), units[0])
        np.save("./prediction results/" + string_name, pred)
        model.save("./model_saves/" + string_name)

#%% load prediction results
dict_results = {}
for output_period in output_periods[4:6]:
    for units in unit_structures:
        string_name = "%dus%dcells%dunits"%(output_period, len(units), units[0])
        pred = np.load("./prediction results/" + string_name+".npy")
        dict_results[output_period, len(units), units[0]] = pred

#%% SNR table

#snr calculated across the entire dataset
snr_table = np.zeros((len(output_periods), len(unit_structures)))
for i in range(len(output_period)):
    sample_period = output_periods[i]*(10**-6)/16
    (X, X_train, X_test), (y, y_train, y_test), \
        (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
    for j in range(len(unit_structures)):
        pred = dict_results[output_periods[i], len(unit_structures[j]),unit_structures[j][0]].T
        true = pin_scaler.inverse_transform(np.expand_dims(y,-1))
        snr = signaltonoise(true, pred)
        snr_table[i,j] = snr

print(snr_table)
np.savetxt("./prediction results/SNR table.csv", snr_table, delimiter=',')
#%% SNR table for only validation profile

snr_table = np.zeros((len(output_periods), len(unit_structures)))
for i in range(len(output_periods)):
    sample_period = output_periods[i]*(10**-6)/16
    (X, X_train, X_test), (y, y_train, y_test), \
        (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
    true = pin_scaler.inverse_transform(np.expand_dims(y_train,-1))
    for j in range(len(unit_structures)):
        units = unit_structures[j]
        string_name = "%dus%dcells%dunits"%(output_periods[i], len(units), units[0])
        print("inferring for "+ string_name)
        model = keras.models.load_model("C:/Users/dncob/Documents/GitHub/LSTM-acceleration-with-singular-value-decomposition/code/model_saves/" + string_name)
        pred = pin_scaler.inverse_transform(model.predict(X_test)[0]).T
        snr = signaltonoise(true, pred)
        snr_table[i,j] = snr

print(snr_table)
np.savetxt("./prediction results/SNR validation table.csv", snr_table, delimiter=',')


#%% save model in ONNX format
import tf2onnx
import onnx
import onnxruntime as rt

sample_period = output_periods[3]*(10**-6)/16
(X, X_train, X_test), (y, y_train, y_test), \
    (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)

model = keras.models.load_model("./model_saves/400us3cells15units")
spec = (tf.TensorSpec((None, None, 16), tf.double, name="input"),)
output_path = "./model_saves/RT-implementation.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13,output_path=output_path)

providers = ['CPUExecutionProvider']
output_names = [n.name for n in model_proto.graph.output]
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": X_test})
onnx_pred = np.array(onnx_pred)
preds = model.predict(X_test)

plt.figure()
plt.plot(onnx_pred.squeeze())

# make sure ONNX and keras have the same results
np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-5)


np.savetxt("preprocessed_DROPBEAR_X.csv", X.squeeze(), delimiter=',');
np.savetxt("preprocessed_DROPBEAR_t.csv", t, delimiter=',')
np.savetxt("preprocessed_DROPBEAR_y.csv", y, delimiter=',')
np.savetxt("model_prediction.csv", onnx_pred.squeeze(), delimiter=',')