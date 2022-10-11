import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from scipy import signal
from sklearn.metrics import mean_squared_error
def signaltonoise(signal, noisy_signal, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
# save model weights as folder of csvs.
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
def save_model_weights_as_csv(model, savpath = "./model_weights"):
    import os
    import os.path as path
    from numpy import savetxt
    if(not path.exists(savpath)):
        os.mkdir(savpath)
    layer_index = 1
    for layer in model.layers[:-1]: # the final layer is a dense top
        layer_path = savpath + "./" + "layer " + str(layer_index) + "./"
        layer_index += 1
        if(not path.exists(layer_path)):
            os.mkdir(layer_path)
        W, U, b = layer.get_weights()
        units = U.shape[0]
        
        savetxt(layer_path+"Wi.csv",W[:,:units].T,delimiter=',')
        savetxt(layer_path+"Wf.csv",W[:,units:units*2].T,delimiter=',')
        savetxt(layer_path+"Wc.csv",W[:,units*2:units*3].T,delimiter=',')
        savetxt(layer_path+"Wo.csv",W[:,units*3:].T,delimiter=',')
        savetxt(layer_path+"Ui.csv",U[:,:units].T,delimiter=',')
        savetxt(layer_path+"Uf.csv",U[:,units:units*2].T,delimiter=',')
        savetxt(layer_path+"Uc.csv",U[:,units*2:units*3].T,delimiter=',')
        savetxt(layer_path+"Uo.csv",U[:,units*3:].T,delimiter=',')
        savetxt(layer_path+"bi.csv",b[:units],delimiter=',')
        savetxt(layer_path+"bf.csv",b[units:units*2],delimiter=',')
        savetxt(layer_path+"bc.csv",b[units*2:units*3],delimiter=',')
        savetxt(layer_path+"bo.csv",b[units*3:],delimiter=',')
    
    #save dense top layer
    dense_top = model.layers[-1]
    in_weights, out_weights = dense_top.get_weights()
    layer_path = savpath + "./dense_top./"
    if(not path.exists(layer_path)):
        os.mkdir(layer_path)    
    savetxt(layer_path+"weights.csv",in_weights,delimiter=',')
    savetxt(layer_path+"bias.csv",out_weights,delimiter=',')

if __name__ == "__main__":
    sample_period = 400*10**-6/16
    # calling preprocess is only necessary when output period changes, but probably not the biggest time drag
    (X, X_train, X_test), (y, y_train, y_test), \
        (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(sample_period)
    
    model = keras.models.load_model("./model_saves/400us3cells15units")
    save_model_weights_as_csv(model, savpath="./model_saves/export_model")
    # model = tf.saved_model.load("./model_saves/400us3cells15units")
    p = model.predict(X.reshape(1, -1, 16)).flatten()
    np.savetxt("./model_saves/export_model/model_output.csv", p, delimiter=',')
    snr = signaltonoise(y, p)
    rmse = mean_squared_error(y, p, squared=False)
    print("untransformed SNR: " + str(snr))
    print("untransformed RMSE: " + str(rmse))
    
    y = pin_scaler.inverse_transform(np.expand_dims(y, -1)).flatten()
    p = pin_scaler.inverse_transform(np.expand_dims(p, -1)).flatten()
    
    snr = signaltonoise(y, p)
    rmse = mean_squared_error(y, p, squared=False)
    print("transformed SNR: " + str(snr))
    print("transformed RMSE: " + str(rmse))
    
    print("transform mean: " + str(pin_scaler.mean_[0]))
    print("transform std: " + str(pin_scaler.scale_[0]))
    
    plt.figure()
    plt.plot(t, y, label="true output")
    plt.plot(t, p, label="pred. output")
    plt.legend(loc=1)
    plt.tight_layout()
    

model = keras.models.load_model("./model_saves/frequency")
# model = tf.saved_model.load("./model_saves/400us3cells15units")

X = np.loadtxt("preprocessed_DROPBEAR_X.csv", delimiter=',')
t = np.loadtxt("preprocessed_DROPBEAR_t.csv", delimiter=',')
y = np.loadtxt("preprocessed_DROPBEAR_y.csv", delimiter=',')

p = model.predict(X)
#%%
model = keras.models.load_model("./model_saves/500us2cells15units", compile=False, custom_objects={'early_stopping': None, 'state_resetter': None})
