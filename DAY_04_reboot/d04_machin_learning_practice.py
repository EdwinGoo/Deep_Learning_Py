import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np

print('iris ','■' * 60)

irisdata = pd.read_csv('data/iris.csv', header=None)
iris_X = irisdata.iloc[:, 0:4]
iris_y = target = label = irisdata[4]
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

# print(np.mean(model.predict(X_test)==y_test))
# print(np.mean(model.predict(X_train)==y_train))

print(model.score(X_test,y_test))
print(model.score(X_train,y_train))

# myflowers = [[5.1,3.2,1.1,0.2],[5.4,3,4.5,1.5]]
myflowers = np.array([5.1,3.2,1.1,0.2]).reshape(1,4)
print(model.predict(myflowers))
print(" ")

print('boston ','■' * 60)

bostondata = load_boston()
boston_X = bostondata.data
boston_y = bostondata.target

bX_train, bX_test, by_train, by_test = train_test_split(boston_X,boston_y)

for n in range(1,12,2) :
    print(n)
    bModel = KNeighborsRegressor(n_neighbors = n)
    bModel.fit(bX_train,by_train)

    print(bModel.score(bX_train,by_train))
    print(bModel.score(bX_test,by_test))

# print(pd.DataFrame(X,columns=bostondata.feature_names))
# print(bostondata.DESCR) # Data Set Charateristics DESCR