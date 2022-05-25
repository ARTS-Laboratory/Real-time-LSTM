import joblib
import pickle
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib as mpl
# from math import isnan
import numpy as np
from numpy.random import randint
import math
import scipy
from scipy import signal
from sklearn.metrics import mean_squared_error
"""
Goals of this script:
    1. Input training data will be vector of last few acceleration datapoints.
This requires modifying the preprocess code, but I'll keep that in this file
    2. Separate different parts of the dataset and use those for validation
    3. Create models under a 300 us constraint. 
"""
#%% preprocess
def preprocess():
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
    
    resample_period = (500/16)*10**-6 # maybe change to 300 us
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
    
    X1 = X[:,t<16]
    y1 = y[t<16]
    t1 = t[t<16]
    
    X2 = X[:,np.logical_and(t>=16,t<30.7)]
    y2 = y[np.logical_and(t>=16, t<30.7)]
    t2 = t[np.logical_and(t>=16, t<30.7)]
    
    X3 = X[:,t>30.7]
    y3 = y[t>30.7]
    t3 = t[t>30.7]
    
    
    return (X, X1, X2, X3), (y, y1, y2, y3), (t, t1, t2, t3), pin_scaler, acc_scaler

def split_train_random(X_train, y_train, batch_size, train_len):
    run_size = X_train.shape[1]
    indices = [randint(0, run_size - train_len) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[0,index:index+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index+train_len] for index in indices]))
    return X_mini, y_mini

# use the formula SNR= (A_signal/A_noise)_rms^2. returned in dB
def signaltonoise(signal, noisy_signal):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    return 10*math.log(snr, 10)
#%% load data
X, y, t, pin_scaler, acc_scaler = preprocess()

#touple of (X, y, t train, X, y, t validate)
training_regiments = (
    (np.append(X[2], X[3], axis=1), np.append(y[2],y[3]), np.append(t[2], t[3]), X[1], y[1], t[1]),
    (np.append(X[1], X[3], axis=1), np.append(y[1],y[3]), np.append(t[1], t[3]), X[2], y[2], t[2]),
    (np.append(X[1], X[2], axis=1), np.append(y[1],y[2]), np.append(t[1], t[2]), X[3], y[3], t[3])
)

unit_structures = (
    [40],
    [23,23],
    [18,18,18]
)


# fill this with nparrays of prediction results. keys (val profile, profile, model size)
dict_results = {}

for unit_pattern in range(3):
    for training_pattern in range(3):
        print("training model #" +str(unit_pattern*3+training_pattern))
        units = unit_structures[unit_pattern]
        training_regiment = training_regiments[training_pattern]
        X_train = training_regiment[0]
        y_train = training_regiment[1]
        X_test = training_regiment[3]
        y_test = training_regiment[4]
        
        X_mini, y_mini = split_train_random(X_train, y_train, 10000, 200)

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
        
        for i in range(1,4):
            pred = pin_scaler.inverse_transform(model.predict(X[i])[0]).T
            dict_results[training_pattern, i, unit_pattern] = pred
        model.save("./model_saves/500us" + str(training_pattern) + str(unit_pattern))

#%% save dict results


#%% plots

cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(3): #across model shapes
    validation_matrix = np.zeros((3,3))
    fig, axes = plt.subplots(3,1, sharex=True, sharey=True, figsize=(9,4))
    for j in range(3): # across validation profile
        for k in range(3): # across prediction profile
            pred = dict_results[(j,k+1,i)].T
            true = pin_scaler.inverse_transform(np.expand_dims(y[k+1],-1))
            snr = signaltonoise(true, pred)
            validation_matrix[j, k] = snr
            c = cc[2] if j == k else cc[1]
            axes[j].plot(t[k+1], pred, c=c)
        reference = pin_scaler.inverse_transform(np.expand_dims(y[0],-1))
        axes[j].plot(t[0], reference, c=cc[0], alpha=.9)
    fig.savefig("./plots/validation profile prediction" + str(i) + ".png", dpi=800)
    plt.figure()
    plt.matshow(validation_matrix.T)
    plt.ylabel("validation profile of model")
    plt.xlabel("prediction profile")
    plt.colorbar()
    plt.savefig("./plots/validation matrix" + str(i) + ".png", dpi=800)