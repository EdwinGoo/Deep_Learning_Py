import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

import d02_2_my_perceptron as d

model = d.perceptron()

iris = pd.read_csv('data/iris.csv', header=0)
data = iris[:100]
X = data.iloc[:,0:4].values.astype('float32')
y = data.variety
y = np.where(y=='Setosa', 1, -1)
 
X_train, X_test, y_train, y_test = train_test_split(X,y)

error_history = model.fit(X_train,y_train,10,0.01)
print(model.w, model.b)

# plt.scatter(data.iloc[:, 0], data.iloc[:, 2], c=y)
# plt.plot(error_history, color='g', marker='o', linestyle='--')

data2 = iris[50:]
X2 = data2.iloc[:,0:4].values.astype('float32')
y2 = data2.variety
y2 = np.where(y2=='Virginica',1,-1)
X_train, X_test, y_train, y_test = train_test_split(X2,y2)

error_history2 = model.fit(X_train,y_train,10,0.01)
print(model.w, model.b)
plt.scatter(data2.iloc[:, 0], data2.iloc[:, 2], c=y2)

# plt.plot(error_history2, color='r', marker='x', linestyle='--')
plt.show()

