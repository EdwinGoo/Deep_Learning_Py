# Day_02_02_MultipleRegression.py
import tensorflow as tf
import numpy as np


def multiple_regression_0():
    w = tf.Variable(tf.random_uniform([3, 5]))
    # w = tf.Variable(tf.random_normal([15]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(sess.run(w))
    sess.close()

    # [[0.85428333 0.44007075 0.52131724 0.5502416  0.84345794]
    #  [0.11354446 0.6419821  0.41817164 0.20478499 0.1825583 ]
    #  [0.5679046  0.09878469 0.5185164  0.58605146 0.42177343]]


# 문제
# 공부한 시간과 출석한 일수를 사용해서 재구성하세요
def multiple_regression_1():
    # y = x1 + x2
    x1 = [1, 0, 3, 0, 5]        # 공부한 시간
    x2 = [0, 2, 0, 4, 0]        # 출석한 일수
    y = [1, 2, 3, 4, 5]

    w1 = tf.Variable(tf.random_uniform([1]))
    w2 = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w1 * x1 + w2 * x2 + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# x1과 x2를 변수 1개로 합쳐보세요
def multiple_regression_2():
    x = [[1, 0, 3, 0, 5],        # 공부한 시간
         [0, 2, 0, 4, 0]]        # 출석한 일수
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([2]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w[0] * x[0] + w[1] * x[1] + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# bias를 없애보세요
def multiple_regression_3():
    # x = [[1, 0, 3, 0, 5],        # 공부한 시간
    #      [0, 2, 0, 4, 0],        # 출석한 일수
    #      [1, 1, 1, 1, 1]]
    # x = [[1, 0, 3, 0, 5],        # 공부한 시간
    #      [1, 1, 1, 1, 1],
    #      [0, 2, 0, 4, 0]]        # 출석한 일수
    x = [[1, 1, 1, 1, 1],
         [1, 0, 3, 0, 5],        # 공부한 시간
         [0, 2, 0, 4, 0]]        # 출석한 일수
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([3]))

    # hx = w[0] * x[0] + w[1] * x[1] + b
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * 1     # b
    hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]    # [b, b, b, b, b]
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(w))
    sess.close()


# 문제
# 행렬 곱셈으로 수정하세요. tf.matmul()
def multiple_regression_4():
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],        # 공부한 시간
         [0., 2., 0., 4., 0.]]        # 출석한 일수
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3]))

    # (1, 5) = (1, 3) @ (3, 5)
    hx = tf.matmul(w, x)
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# 행렬 곱셈에서 w와 x 위치를 바꾸세요
def multiple_regression_5():
    x = [[1., 1., 0.],
         [1., 0., 2.],
         [1., 3., 0.],
         [1., 0., 4.],
         [1., 5., 0.]]
    y = [[1], [2], [3], [4], [5]]

    # x = [[1., 1., 1., 1., 1.],
    #      [1., 0., 3., 0., 5.],        # 공부한 시간
    #      [0., 2., 0., 4., 0.]]        # 출석한 일수
    # x = np.transpose(x)
    # x = np.float32(x)

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (5, 1) = (5, 3) @ (3, 1)
    hx = tf.matmul(x, w)
    # hx = tf.matmul(x, w, transpose_a=True)
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# multiple_regression_0()
# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
multiple_regression_5()








