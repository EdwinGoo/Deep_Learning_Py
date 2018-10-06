# Day_01_01_python.py

# ctrl + shift + f10
# alt + 1
# alt + 4
# shift + arrow
# ctrl + /

# 연산 : 산술, 관계, 논리
# 산술 : +  -  *  /(실수 나눗셈)   **(지수)  //(정수 나눗셈)  %
# 관계 : >  >=  <  <=  ==  !=
# 논리 : and  or  not

a = 13
if a % 2:
    print('홀수')
else:
    print('짝수')

for i in range(0, 5, 1):
    print(i, end=' ')
print()

for i in range(0, 5):
    print(i, end=' ')
print()

for i in range(5):
    print(i, end=' ')
print()
print('-' * 50)


def f_1(a1, a2, a3):
    print(a1, a2, a3)


c = f_1(1, 2, 3)
print(c)


def f_2(a1, a2, a3):
    return a1 + a2 + a3


d = f_2(1, 2, 3)
print(d)


def f_3(a1, a2, a3):
    print(a1, a2, a3)


f_3(1, 2, 3)            # positional
f_3(a1=1, a2=2, a3=3)   # keyword
f_3(1, 2, a3=3)
print('-' * 50)

# collection : list, tuple, set, dictionary
#               []    ()          {}
a = [1, 3, 5]
print(a)
print(a[0], a[1], a[2])

a.append(7)

for i in range(len(a)):
    print(i, a[i])

for i in a:
    print(i, end=' ')
print()

b = (1, 4, 7)
print(b)
print(b[0], b[1], b[2])

# b.append(9)       # 변경 불가
print('-' * 50)

a1, a2 = 3, 6
print(a1, a2)

a3 = 3, 6
print(a3)

a4 = a3
print(a4)

a5, a6 = a3
print(a5, a6)

b = 1, 4, 7
print(b)
print('-' * 50)


def f_4(a1, a2, a3):
    return a1 + a2 + a3, a1 * a2 * a3


a, b = f_4(1, 3, 5)
print(a, b)

c = f_4(1, 3, 5)
print(c)

d, _ = f_4(1, 3, 5)
print(d)
print('-' * 50)

k = 'keyword'

#     key     value
d = {'name': 'hoon', 'age': 20, 3: 14, k: 'hello'}
print(d)
print(d['name'], d['age'], d[3])
print(d[k], d['keyword'])
