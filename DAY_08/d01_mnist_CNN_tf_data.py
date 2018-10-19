import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.losses import categorical_crossentropy

from tensorflow.examples.tutorials.mnist import input_data
import pickle

mnist = input_data.read_data_sets('mnist', one_hot=True, reshape=False)
X_train= mnist.train.images
Y_train= mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

model = Sequential()
# padding을 줌으로 기존 데이터의 shape을 유지해준다.
model.add(Conv2D(20, kernel_size=(5,5), input_shape=(28,28,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(50,kernel_size=5, activation='relu',  padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss=categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=20, 
                    batch_size=100, validation_split=0.2)