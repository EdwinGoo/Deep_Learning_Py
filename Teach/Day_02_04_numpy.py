# Day_02_04_numpy.py
import numpy as np

a1 = np.arange(3)
a2 = np.arange(6)
a3 = np.arange(3).reshape(1, 3)
a4 = np.arange(3).reshape(3, 1)
a5 = np.arange(6).reshape(2, 3)

# print(a1 + 1)         # broadcast
# print(a1 + a1)        # vector

# print(a1 + a2)        # error
print(a1 + a3)          # vector
print(a1 + a4)          # broadcast + broadcast
print(a1 + a5)          # broadcast + vector

# print(a2 + a3)        # error
print(a2 + a4)          # broadcast + broadcast
# print(a2 + a5)        # error

print(a3 + a4)          # broadcast + broadcast
print(a3 + a5)          # broadcast + vector

# print(a4 + a5)        # error
print('-' * 50)

np.random.seed(12)
a = np.random.randint(0, 100, 12).reshape(3, 4)
print(a)

print(np.sum(a))
print(np.sum(a, axis=0))        # 수직(열, column)
print(np.sum(a, axis=1))        # 수평(행, row)

print(np.max(a))
print(np.max(a, axis=0))        # 수직(열, column)
print(np.max(a, axis=1))        # 수평(행, row)

print(np.argmax(a))
print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))
print('-' * 50)

b = np.arange(12)
print(b)

print(b[2], b[7])
print(b[[2, 7, 3]])         # index array

c = np.reshape(b, newshape=[3, 4])
print(c, end='\n\n')

# 문제
# 2차원 배열을 거꾸로 출력하세요
print(c[::-1], end='\n\n')
print(c[::-1][::-1], end='\n\n')

print(c[::-1, ::-1], end='\n\n')    # fancy indexing

print(c[0][0], c[-1][-1])
print(c[0, 0], c[-1, -1])

c[0] = 99
print(c)
print('-' * 50)

# 문제
# 테두리가 1로, 속은 0으로 채워진 5x5 배열을 만드세요
z1 = np.zeros([5, 5], dtype=np.int32)

# z1[0], z1[-1] = 1, 1
# z1[:, 0], z1[:, -1] = 1, 1

z1[[0, -1]] = 1
z1[:, [0, -1]] = 1

print(z1, end='\n\n')

z2 = np.ones([5, 5], dtype=np.int32)

z2[1:-1, 1:-1] = 0

print(z2, end='\n\n')
print('-' * 50)

d = np.arange(6).reshape(2, 3)
print(d, end='\n\n')

print(d.T, end='\n\n')

# 문제
# 2차원 배열을 팬시인덱싱과 반복문을 사용해서
# 전치 형태로 출력하세요.
for i in range(d.shape[-1]):
    print(i, d[:, i])
