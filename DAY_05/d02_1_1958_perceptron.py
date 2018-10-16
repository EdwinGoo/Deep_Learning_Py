import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

class 뉴런:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y = np.where(z > 0, 1, -1)
        return y
        
class 퍼셉트론(뉴런):
    def __init__(self) :
        뉴런.__init__(self, w=None, b=None)

    def fit(self, X,y,학습횟수, 학습률) :

        특징수 = X.shape[1]
        self.w = np.zeros(특징수)
        self.b = 0.

        error_history = []
        for _ in range(학습횟수) :
            종합오류 = 0
            for xi, yi in zip(X,y):
                y_predict = self.predict(xi)
                error = yi-y_predict
                종합오류 += error**2
                delta = error * 학습률
                self.w += delta * xi
                self.b += delta 
            error_history.append(종합오류)
        return error_history