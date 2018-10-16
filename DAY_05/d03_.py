import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

import d02_1958_perceptron as d

iris = pd.read_csv('data/iris.csv', header=None)
data = iris[:100]
X = data.iloc[:,0:4].values.astype('float32')
y = data[4]

y = np.where(y=='Setosa', 1, -1)
X_train, X_test, y_train, y_test = train_test_split(X,y)

model = d.퍼셉트론()

eh = model.fit(X_train,y_train,30,0.01)
print(model.w, model.b)

plt.plot(eh, color='g', marker='o', linestyle='--')
plt.show()