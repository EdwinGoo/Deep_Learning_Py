import numpy as np  

# normal double
normdata = list(range(10))
print(normdata)
doubledata = [n * 2 for n in normdata ]
print(doubledata)

# numpy double
npdata = [[0,1],[1,0]]
# print(type(arraydata)) ndarray
arraydata = np.array(npdata)
print(arraydata)
print(arraydata*2)

#t