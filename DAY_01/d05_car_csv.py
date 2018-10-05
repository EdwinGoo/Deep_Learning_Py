
import csv
#선형회귀
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_car_data() :
    f = open('./data/cars.csv', 'r', encoding='utf-8')
    readfile = csv.reader(f)

    speed , dist= [], []

    for x,y in readfile :
        speed.append(int(x))
        dist.append(int(y))
    f.close() 

    return speed, dist

def get_car_data2() :
    f = open('./data/cars.csv', 'r', encoding='utf-8')
    f.readline() # 헤더 건너뛰기 꼼수

    readfile = csv.reader(f)
    speed , dist= [], []

    for x,y in readfile :
        speed.append(int(x))
        dist.append(int(y))
    f.close() 

    return speed, dist

def get_car_data3() :
    return np.loadtxt('./data/cars.csv', delimiter=',', unpack=True)

def linear_regression_3() :

    xx, y = get_car_data3()

    w = tf.Variable(10.0) # 타입에러 있을 수 있다 가중치랑 바이어스는 실수이므로 10이 아니라 10.0
    b = tf.Variable(10.)

    x = tf.placeholder(tf.float32)

    hx = w * x + b
    y = tf.placeholder(tf.float32)
    loss = tf.reduce_mean((hx-y)**2)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x:xx})
        # print(i, sess.run(loss, {x:xx}))

    y0 = sess.run(hx, {x: 0})
    y1 = sess.run(hx, {x: 30})
    y2 = sess.run(hx, {x: 50})
    
    print(y1,y2)
    sess.close()

    plt.plot(xx, y, 'ro')

    # 문제
    # 우리가 예측한 회귀선을 그려보세요
    plt.plot([0, 30], [0, y1], 'r')
    plt.plot([0, 30], [y0, y1], 'g')

    plt.show()

linear_regression_3()