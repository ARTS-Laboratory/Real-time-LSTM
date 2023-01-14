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
load the model and perform a prediction

Tensorflow 2.5.0
numpy 1.19.5
"""
X = np.load("./dataset/X.npy")
y = np.load("./dataset/y.npy")
t = np.array([1/400*i for i in range(X.shape[1])])
#%%
model = keras.models.load_model("./model")

y_pred = model.predict(X)
#%% plot results
plt.figure(figsize=(6,2.5))
plt.plot(t, y.flatten(), label='reference')
plt.plot(t, y_pred.flatten(), label='prediction')
plt.xlabel("time (s)")
plt.ylabel("roller location (m)")
plt.xlim((0,t[-1]))
plt.legend(loc=2, fancybox=False, framealpha=1)
plt.grid()
plt.tight_layout()