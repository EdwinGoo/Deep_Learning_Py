import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.losses import categorical_crossentropy

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = pd.get_dummies(y_train.flatten()).values
Y_test = pd.get_dummies(y_test.flatten()).values

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['acc'])
history = model.fit(X_train, Y_train, epochs=20, batch_size=100, validation_split=0.2)


# result = pd.DataFrame(history.history)
# result.to_pickle("./img_cnn.pkl")
# model.save('img_cnn.h5')
# del model  # deletes the existing model
