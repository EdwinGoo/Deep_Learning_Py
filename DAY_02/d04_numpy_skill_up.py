# Day_02_04_numpy.py
import numpy as np

a1 = np.arange(3)
a2 = np.arange(6)
a3 = np.arange(3).reshape(1, 3)
a4 = np.arange(3).reshape(3, 1)
a5 = np.arange(6).reshape(2, 3)

print(a1 + 1)         # broadcast
print('#' * 30)
print(a1 + a1)        # vector
print('#' * 30)
# print(a1 + a2)        # error
print(a1 + a3)          # vector
print('#' * 30)
print(a1 + a4)          # broadcast + broadcast
print('#' * 30)
print(a1 + a5)          # broadcast + vector
print('#' * 30)
# print(a2 + a3)        # error
print(a2 + a4)          # broadcast + broadcast
print('#' * 30)
# print(a2 + a5)        # error
print(a3 + a4)          # broadcast + broadcast
print('#' * 30)
print(a3 + a5)          # broadcast + vector
# print(a4 + a5)        # error
print('~(^-^)~' * 20)

np.random.seed(12)
a = np.random.randint(0,100,12).reshape(3,4)

# [[75 27  6  2]
#  [ 3 67 76 48]
#  [22 49 52  5]]
print(a)

print(np.sum(a))
print(np.sum(a,axis=0)) # axis = 0 열
print(np.sum(a,axis=1)) # axis = 1 행

print(np.max(a))
print(np.max(a,axis=0))
print(np.max(a,axis=1))

print(np.argmax(a))
print(np.argmax(a,axis=0)) # [0,1,1,1] 75가 크다 0번째 67이 크다 1번째 76이 크다 1번째 48이 크다 1번째
print(np.argmax(a,axis=1))