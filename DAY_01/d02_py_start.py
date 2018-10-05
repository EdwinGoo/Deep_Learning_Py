import numpy as np
import tensorflow

a = np.arange(6)
print(a)
print(type(a))
print(a.shape, a.dtype)

b = a.reshape(2,3)
print(b)
print(type(b))
print(b.shape, b.dtype)


x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print(x)

print(x[1,2])
y = x[:,0]
print(y)

print(a[:3])
print(a[3:])
################################   :: ################### 
print(a[::2])
print(a[1::2])
################################

## 역배열 ##
print(a[::-1])
print(list(reversed(a)))

c = np.arange(6)
print(c.reshape(-1,3))
k=np.reshape(c,(2,3),order='A')
print(k)

print(c[False])
print(c[c>3])

print(np.sin(c))

d = np.arange(0,540,90)
print(np.sin(np.reshape(d,(2,3))))

z = np.zeros(10,dtype="int32")  ## default flaot64 
print(z.dtype)
z = np.zeros(10,dtype=np.float32)  ## default flaot64 
print(z.dtype)

x = np.random.randint(9, size=(3, 3))
print(x)
x.itemset(4, 0)
x.itemset((2, 2), 9)
print(x)