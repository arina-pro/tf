#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf #Подключаем модуль Тензорфлоу
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_dim=2),
# Add another:
layers.Dense(1, activation='sigmoid')])


model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

import numpy as np

# the four different states of the XOR gate
data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
labels = np.array([[0],[1],[1],[0]], "float32")

model.fit(data, labels, epochs=50, batch_size=1)

model.evaluate(data, labels, batch_size=1)

print(model.predict(data, batch_size=1))
