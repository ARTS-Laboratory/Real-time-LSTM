import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


model = keras.models.load_model("./model_saves/400us3cells15units")
# model = tf.saved_model.load("./model_saves/400us3cells15units")

X = np.loadtxt("preprocessed_DROPBEAR_X.csv", delimiter=',')
t = np.loadtxt("preprocessed_DROPBEAR_t.csv", delimiter=',')
y = np.loadtxt("preprocessed_DROPBEAR_y.csv", delimiter=',')

model.predict(X)

