# Day_02_03_MultiplyRegression_trees.py
import tensorflow as tf
import numpy as np


# 문제
# Girth와 Height가 15, 75일 때와
# Girth와 Height가 20, 85일 때의 Volume을 알려주세요.
# (placeholder 사용)
# 14.5,74,36.3
# 20.6,87,77
def multiple_trees_1():
    trees = np.loadtxt('Data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    print(trees.shape)
    print(trees[0])

    # x = np.array([trees[0], trees[1]])
    x = trees[:-1]
    y = trees[-1]

    w = tf.Variable(tf.random_uniform([1, 2]))
    b = tf.Variable(tf.random_uniform([1]))

    # (1, 31) = (1, 2) @ (2, 31)
    hx = tf.matmul(w, x) + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


def multiple_trees_2():
    girth, height, volume = np.loadtxt('Data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)

    # xx = [[1] * 31, girth, height]
    xx = np.float32([np.ones(31), girth, height])
    y = volume

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([1, 3]))

    # (1, 31) = (1, 3) @ (3, 31)
    hx = tf.matmul(w, x)
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(loss, {x: xx}))
    print('-' * 50)

    # (1, 1) = (1, 3) @ (3, 1)
    print(sess.run(hx, {x: [[1], [15], [75]]}))
    print(sess.run(hx, {x: [[1, 1],
                            [15, 20],
                            [75, 85]]}))

    sess.close()


# 문제
# loadtxt 함수 호출할 때 unpack 옵션을 사용하지 말고 처리하세요
# 팬시 인덱싱을 사용합니다
def multiple_trees_3():
    trees = np.loadtxt('Data/trees.csv', delimiter=',', dtype=np.float32)

    # xx = [[1] * 31, girth, height]
    xx = np.float32([np.ones(31), trees[:, 0], trees[:, 1]])
    y = trees[:, -1]

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([1, 3]))

    # (1, 31) = (1, 3) @ (3, 31)
    hx = tf.matmul(w, x)
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(loss, {x: xx}))
    print('-' * 50)

    # (1, 1) = (1, 3) @ (3, 1)
    print(sess.run(hx, {x: [[1], [15], [75]]}))
    print(sess.run(hx, {x: [[1, 1],
                            [15, 20],
                            [75, 85]]}))

    sess.close()


def multiple_trees_4():
    trees = np.loadtxt('Data/trees.csv', delimiter=',', dtype=np.float32)

    xx = trees[:, :-1]
    # y = trees[:, -1].reshape(-1, 1)
    y = trees[:, -1:]
    print(xx.shape, y.shape)

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([2, 1]))
    b = tf.Variable(tf.random_uniform([1]))

    # (31, 1) = (31, 2) @ (2, 1)
    hx = tf.matmul(x, w) + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(loss, {x: xx}))
    print('-' * 50)

    print(sess.run(hx, {x: [[15, 75],
                            [20, 85]]}))

    sess.close()


# multiple_trees_1()
# multiple_trees_2()
# multiple_trees_3()
multiple_trees_4()
