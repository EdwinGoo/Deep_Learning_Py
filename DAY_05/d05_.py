import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def sigmoid(x) :
    y = 1/(1+np.exp(-x))
    return y

x = np.array([1.,0.5])
print(x.shape)

w1 = np.array([[0.1,0.1,0.2],[0.2,0.3,0.1]])
z1 = np.dot(x,w1)
a1 = sigmoid(z1)
print(a1)

w2 = np.array([[[0.1,-0.4],[0.2,0.3],[0.4,0.1]]])
z2 = np.dot(a1,w2)
a2 = sigmoid(z2)
print(a2)

w3 = np.array([[0.7,0.2],[0.7,0.1]])
z3 = np.dot(a2,w3)
a3 = sigmoid(z3)
print(a3)

