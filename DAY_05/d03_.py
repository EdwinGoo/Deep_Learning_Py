import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

import d02_2_my_perceptron as d

iris = pd.read_csv('data/iris.csv', header=None)
data = iris[:100]
X = data.iloc[:,0:4].values.astype('float32')
y = data[4]
y = np.where(y=='Setosa', 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = d.perceptron()
 
error_history = model.fit(X_train,y_train,30,0.01)
print(model.w, model.b)

# plt.plot(error_history, color='g', marker='o', linestyle='--')
# plt.show()

y_pred = model.activation(X_test)
print(np.mean(y_pred == y_test))

data2 = iris[50:]
X2 = data2.iloc[:,0:4].values.astype('float32')
y2 = data[4]
y2 = np.where(y=='Virginica',1,-1)
X_train, X_test, y_train, y_test = train_test_split(X2,y2)
