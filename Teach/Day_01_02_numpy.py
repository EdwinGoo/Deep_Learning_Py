# Day_01_02_numpy.py
import numpy as np

a = np.arange(6)
print(a)
print(type(a))
print(a.shape, a.dtype)

b = a.reshape(2, 3)
print(b)
print(b.shape, b.dtype)
print('-' * 50)

print(a[0], a[1], a[len(a) - 1], a[-1])
print(b[0], b[1])

print(a[2:5])       # slicing, range()

# 문제
# 앞쪽 절반을 출력하세요
# 뒤쪽 절반을 출력하세요
print(a[0:len(a)//2])
print(a[0:3])
print(a[:3])
print()

print(a[len(a)//2:len(a)])
print(a[3:6])
print(a[3:])
print()

# 문제
# 짝수 번째만 출력하세요
# 홀수 번째만 출력하세요
print(a[::2])
print(a[1::2])
print()

# 문제
# 거꾸로 출력하세요
print(a[3:4])
print(a[3:3])
print(a[5:0:-1])
print(a[5:-1:-1])
print(a[-1:-1:-1])
print(a[::-1])      # 음수 증감(역방향)
print('-' * 50)

c = np.arange(6)

c.reshape(2, 3)
# c = c.reshape(2, 3)
print(c)

print(c.reshape(2, 3))
print(c.reshape(-1, 3))
print(c.reshape(2, -1))
print('-' * 50)

print(a)
print(a + 1)            # broadcast
print(a ** 2)
print(a > 1)
print(a[a > 1])         # boolean index
print(np.sin(a))        # universal function
print()

print(b)
print(b + 1)            # broadcast
print(b ** 2)
print(b > 1)
print(b[b > 1])         # boolean index
print(np.sin(b))        # universal function
print()

print(a + a)            # vector operation
print(b + b)
print()

e1 = np.zeros(3)
print(e1, e1.dtype)

e2 = np.zeros(3, dtype=np.float32)
print(e2, e2.dtype)
