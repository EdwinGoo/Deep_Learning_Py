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
print(a)



