import numpy as np

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
