import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.image import imread

import pickle
with open('data/mnist_weight.pkl', 'rb') as 파일:
    params = pickle.load(파일)
params.keys()

mnist = input_data.read_data_sets('mnist', one_hot=True)
X_test = mnist.test.images
Y_test = mnist.test.labels
print(X_test.max(), X_test.min())

def sigmoid(x) : 
    return 1/(1+np.exp(-x))

class Layer() :
    def __init__(self, input, output, activation) :
        self.W = np.random.randn(input, output)
        self.b = np.random.randn(input)
        self.active = activation
    def forward(self,X) :
        z = np.dot(X, self.W) + self.b
        return self.active(z)

class FeedForwardNet() :
    def __init__(self) :
        self.layers = []
    def addLayer(self,layer) :
        self.layers.append(layer)
    def predict(self, X) :
        layer_output = X
        for layer in self.layers :
            layer_output = layer.forward(layer_output)
        y = layer_output
        return y

model = FeedForwardNet()

layer1 = Layer(784, 50, sigmoid)
layer2 = Layer(50, 100, sigmoid)
layer3 = Layer(100, 10, lambda x:x)

model.addLayer(layer1)
model.addLayer(layer2)
model.addLayer(layer3)

model.layers[0].W = params['W1']
model.layers[0].b = params['b1']
model.layers[1].W = params['W2']
model.layers[1].b = params['b2']
model.layers[2].W = params['W3']
model.layers[2].b = params['b3']

Y_pred = model.predict(X_test)
print(Y_pred.shape)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred[:5])
Y_predtest = np.argmax(Y_test, axis=1)
print(Y_predtest[:5])