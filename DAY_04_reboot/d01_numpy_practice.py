import numpy as np  
import matplotlib.pyplot as plt
# normal double
normdata = list(range(10))
print(normdata)
doubledata = [n * 2 for n in normdata ]
print(doubledata)

# numpy double
npdata = [[0,1],[1,0],[1,1]]
# print(type(arraydata)) ndarray
arraydata = np.array(npdata)
print(arraydata)
print(arraydata*2)

arraydata = arraydata.reshape(2,3)
print(arraydata)
arraydata=arraydata.flatten()
print(arraydata)

ninedata = np.arange(9)
print(ninedata)
print(ninedata.shape)
ninedata = ninedata.reshape(9,1)
print(ninedata)
print(ninedata.shape)
bigdata = ninedata*np.array([1,2,3])
print(bigdata)
# 행렬의 데이터를 선택했지만, 벡터로 표현됨
print(bigdata[:,0])
print(bigdata[1:3,1:3])

print(bigdata.sum(axis=0)) #sum(axis=0) sum of each column  
print(bigdata.max(axis=0)) #sum(axis=0) max of each column  

#fanxy index
print(bigdata[[0,3]])

#boolean index
testbool = np.arange(9)
print(testbool)
boolfilter = testbool>4
print(testbool[testbool>5])
print(testbool[boolfilter])

testbool = testbool.reshape(3,3)
boolfilter = testbool>4
print(testbool)
print(testbool[boolfilter])

testbool = np.where(testbool>5,-1,testbool) # where(조건, Treu Value, False Value)
print(testbool)

zerodata = np.ones((5,5))
zerodata[1:-1, 1:-1] = 0
print(zerodata)

# suffle vs permutation
x = np.arange(10)
np.random.shuffle(x)
print(x)
np.random.permutation(x)
print(x)
x = np.random.permutation(x)
print(x)

# random
# uniformrand = np.random.uniform(low=0,high=2,size=3)
uniformrand = np.random.uniform(low=0,high=2,size=(5,3))
print(uniformrand)
print(np.random.random((3,4)))
# normal 
normrand = np.random.normal(0,1,(4,4))
print(normrand)

x = np.arange(0, 6, 0.1)
y = np.sin(x)
y2 = np.cos(x)
plt.plot(x,y, color = 'r' , marker = 'x')
plt.plot(x,y2)
plt.show()

# 1. 9개의 numpy 벡터 생성하기
dataset1 = np.arange(9)
print(dataset1)
print(type(dataset1))
print(dataset1.shape)

# 2. 2x2의 행렬 생성하기
dataset2 = np.array([(0,1),(1,1)])
print(dataset2)
dataset3 = np.arange(4).reshape(2,2)
print(dataset3)