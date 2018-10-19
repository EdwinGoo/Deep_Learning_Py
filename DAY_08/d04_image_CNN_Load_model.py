import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

from keras.datasets import cifar10
from keras.models import load_model

from scipy.misc import imresize 

# mozzi = imread('data/mozzi.jpg')
mozzi = imread('data/santafe.jpg')
mozzi = imresize(mozzi, (32,32,3))
# plt.imshow(mozzi)
# plt.show()
print(mozzi.shape)
mozziNew = np.array([mozzi])

mozzi = mozzi.astype('float32')/255. #모찌는 640 640 3 , 우리 모델은 32,32,3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_test = X_test.astype('float32')/255.
X_train = X_train.astype('float32')/255.

Y_train = pd.get_dummies(y_train.flatten()).values
Y_test = pd.get_dummies(y_test.flatten()).values

model = load_model('DAY_08/models/cifar-10_cnn_deep.h5')
model.summary()

# scores = model.evaluate(X_test,Y_test)
# print('Loss: {:.3f}, Acc: {:.3f}'.format(*scores))

Y_pred = model.predict(mozziNew) 
print(Y_pred)
print(np.argmax(Y_pred,axis=1))