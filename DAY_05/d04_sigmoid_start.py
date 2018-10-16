import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

x = np.arange(-5,5,0.1)

def step(x):
    y = np.where(x > 0, 1, -1)
    return y

def sigmoid(x) :
    y = 1/(1+np.exp(-x))
    return y

plt.plot(x, step(x))
plt.plot(x, sigmoid(x))
plt.show()
