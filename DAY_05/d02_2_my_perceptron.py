# 1958

import numpy as np
import pandas as pd

class cellbody :
    def __init__(self, w, b) : 
        self.w = w
        self.b = b
    def activation(self, X) :
        z = np.dot(X, self.w) + self.b
        y = np.where(z>0, 1, -1)
        return y

class perceptron(cellbody) :
    def __init__(self) :
        cellbody.__init__(self, w=None, b=None)
    def fit(self, X, y, cycle, rate) :
        fearture = X.shape[1]
        self.w = np.zeros(fearture)
        self.b = 0.

        error_history = []

        for _ in range(cycle) :
            errorsum = 0
            for xi, yi in zip(X,y) :
                y_predict = self.activation(xi)
                error = yi - y_predict
                errorsum += error**2
                self.w += error * rate * xi
                self.b += error * rate

            error_history.append(errorsum)
        return error_history
