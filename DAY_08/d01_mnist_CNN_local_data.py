import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.losses import categorical_crossentropy

from mnist_local_data.dataset import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_mnist(flatten=False, normalize=True, one_hot_label=True)
X_train = np.transpose(X_train, [0, 2, 3, 1])
X_test = np.transpose(X_test, [0, 2, 3, 1])
print(X_train.shape)

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