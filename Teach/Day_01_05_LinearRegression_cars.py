# Day_01_05_LinearRegression_cars.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_cars_1():
    cars = np.loadtxt('Data/cars.csv', delimiter=',')
    # print(cars)

    speed, dist = [], []
    for s, d in cars:
        # print(s, d)
        speed.append(s)
        dist.append(d)

    return speed, dist


def get_cars_2():
    f = open('Data/cars.csv', 'r', encoding='utf-8')

    # skip header.
    f.readline()

    speed, dist = [], []
    for line in f:
        print(line.strip().split(','))
        s, d = line.strip().split(',')
        speed.append(int(s))
        dist.append(int(d))

    return speed, dist


def get_cars_3():
    # cars = np.loadtxt('Data/cars.csv', delimiter=',', unpack=True)
    # print(cars)
    return np.loadtxt('Data/cars.csv', delimiter=',', unpack=True)


# xx, y = get_cars_1()
# xx, y = get_cars_2()
xx, y = get_cars_3()

w = tf.Variable(10.)
b = tf.Variable(10.)

x = tf.placeholder(tf.float32)

hx = w * x + b
loss = tf.reduce_mean((hx - y) ** 2)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, feed_dict={x: xx})
    print(i, sess.run(loss, {x: xx}))
print('-' * 50)

# 문제
# speed가 30과 50일 때의 제동거리를 알려주세요
y0 = sess.run(hx, {x: 0})
y1 = sess.run(hx, {x: 30})
y2 = sess.run(hx, {x: 50})

print(y1, y2)
sess.close()

plt.plot(xx, y, 'ro')

# 문제
# 우리가 예측한 회귀선을 그려보세요
plt.plot([0, 30], [0, y1], 'r')
# plt.plot([0, 30], [y0, y1], 'g')

# speed = [0, 30]
# plt.plot(speed, sess.run(hx, {x: speed}), 'g')
plt.show()
