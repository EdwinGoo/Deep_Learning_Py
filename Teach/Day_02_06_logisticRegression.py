# Day_02_06_logisticRegression.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def show_sigmoid():
    def sigmoid(z):
        return 1 / (1 + np.e ** -z)

    print(np.e)
    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))

    for z in np.linspace(-5, 5, 50):
        s = sigmoid(z)
        print(z, s)

        plt.plot(z, s, 'ro')
    plt.show()


def select():
    def log_a():
        return 'A'

    def log_b():
        return 'B'

    y = 1
    print(y * log_a() + (1 - y) * log_b())

    if y == 1:
        print(log_a())
    else:
        print(log_b())

    y = 0
    print(y * log_a() + (1 - y) * log_b())

    if y == 1:
        print(log_a())
    else:
        print(log_b())


def logistic_regression_1():
    x = [[1., 1., 1., 1., 1., 1.],
         [1., 1., 2., 3., 4., 6.],        # 공부한 시간
         [1., 3., 2., 5., 4., 2.]]        # 출석한 일수
    y = [0, 0, 0, 1, 1, 1]
    y = np.int32(y)

    w = tf.Variable(tf.random_uniform([1, 3]))

    z = tf.matmul(w, x)
    hx = 1 / (1 + tf.exp(-z))
    loss_i = y * -tf.log(hx) + (1 - y) * -tf.log(1 - hx)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    print(sess.run(hx))
    sess.close()


# 문제
# 3시간 공부하고 4번 출석한 학생과
# 5시간 공부하고 2번 출석한 학생의 통과 여부를 알려주세요
def logistic_regression_2():
    xx = [[1., 1., 1., 1., 1., 1.],
          [1., 1., 2., 3., 4., 6.],        # 공부한 시간
          [1., 3., 2., 5., 4., 2.]]        # 출석한 일수
    y = [[0, 0, 0, 1, 1, 1]]
    y = np.float32(y)

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([1, 3]))

    # (1, 6) = (1, 3) @ (3, 6)
    z = tf.matmul(w, x)
    hx = tf.nn.sigmoid(z)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: xx})
        print(i, sess.run(loss, {x: xx}))
    print('-' * 50)

    pred = sess.run(hx, {x: [[1., 1.],
                             [1., 4.],
                             [5., 1.]]})
    print(pred)
    print(pred > 0.5)
    print('-' * 50)

    pred = sess.run(z, {x: [[1., 1.],
                            [1., 4.],
                            [5., 1.]]})
    print(pred)
    print(pred > 0)
    sess.close()


# show_sigmoid()
# select()

# logistic_regression_1()
logistic_regression_2()
