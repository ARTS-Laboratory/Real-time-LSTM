import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
"""
This script trains a 2 cell, 15 unit LSTM model using the data in ./dataset

TensorFlow 2.5.0
Numpy 1.19.5
"""
#%% function definitions
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
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        
        return rtrn[:-1], rtrn[-1]

"""
Early stopping callback with 
"""
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


#%% load dataset
X = np.load("./dataset/X.npy")
X_train = np.load("./dataset/X_train.npy")
X_test = np.load("./dataset/X_test.npy")
y = np.load("./dataset/y.npy")
y_train = np.load("./dataset/y_train.npy")
y_test = np.load("./dataset/y_test.npy")
t = np.array([1/400*i for i in range(X.shape[1])])
#%% define model, training_generator, and callbacks
units = 15
model = keras.Sequential([
    keras.layers.LSTM(units, return_sequences=True, input_shape=[None, 16]), 
    keras.layers.LSTM(units, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
)
early_stopping = DelayEarlyStopping(
    start_epoch=30,
    monitor="val_loss",
    # min_delta=0.001,
    patience=10,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)

X_mini, y_mini = split_train_random(X_train, y_train, 10000, 400)
#%% fit model and save
model.fit(
    X_mini, y_mini,
    batch_size=32,
    shuffle=False,
    epochs=1000,
    validation_data = (X, y.reshape(1, -1, 1)),
    callbacks = [early_stopping],
)

model.save("./model")
#%% perform an inferrence and plot the results
y_pred = model.predict(X)

plt.figure(figsize=(6,2.5))
plt.plot(t, y.flatten(), label='reference')
plt.plot(t, y_pred.flatten(), label='prediction')
plt.xlabel("time (s)")
plt.ylabel("roller location (m)")
plt.xlim((0,t[-1]))
plt.legend(loc=2, fancybox=False, framealpha=1)
plt.grid()
plt.tight_layout()