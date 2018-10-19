import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

from keras.datasets import cifar10
from scipy.signal import correlate # 합성곱 연산

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = pd.get_dummies(y_train.flatten()).values
Y_test = pd.get_dummies(y_test.flatten()).values
# plt.imshow(X_train[0])
# plt.show()

mozzi = imread('data/mozzi.jpg')
plt.show()

kernel = np.array([-1, 1])
kernel = np.expand_dims(np.expand_dims(kernel,-1),0)

edges = correlate(mozzi, kernel, mode='same')
print(edges[:,:,0])
print(mozzi[:,:,0])

edges = np.abs(edges)
edges = edges.astype('uint8')
plt.imshow(edges*10)
plt.show()
