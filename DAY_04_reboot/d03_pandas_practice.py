import pandas as pd

iris = pd.read_csv('data/iris.csv', header=None)
print(type(iris))
print(iris[0:5])

print(iris.iloc[:3,:3])
print(iris.loc[0,4])

# print(iris.values[0:3])
# print('■' * 60)
# print(iris.iloc[:3, 0:4])
# iristype = type(iris.values[0:3])
# # print(iristype)

# label = iris[4]
# print(type(label))
# print(label.value_counts())

