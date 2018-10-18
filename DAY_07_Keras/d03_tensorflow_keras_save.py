from keras.layers import Dense
from keras.models import Sequential
import keras
from keras.models import load_model

import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import pickle

mnist = input_data.read_data_sets('mnist', one_hot=True)
X_train= mnist.train.images
Y_train= mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

# keras를 이용한 NN 구성 기초
model = Sequential()
# model.add(Dense(50, input_shape=(784,), activation='sigmoid'))
model.add(Dense(50, input_shape=(784,), activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

# Epoch 1/200 44000/44000 [==============================] 
# 1s 32us/step - loss: 2.3029 - acc: 0.1212 - val_loss: 2.2789 - val_acc: 0.1529
# ...............................................................................
#  Epoch 200/200 44000/44000 [==============================] 
#  1s 13us/step - loss: 0.1704 - acc: 0.9511 - val_loss: 0.1776 - val_acc: 0.9515
# Train//
history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.2) 

# SAVE 
result = pd.DataFrame(history.history)
result.to_pickle("./add_layer.pkl")
model.save('add_layer.h5')
del model  # deletes the existing model
