# Day_02_01_cost_graph_tf.py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2

    return c / len(x)


def show_cost():
    x = [1, 2, 3]
    y = [1, 2, 3]

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)

        plt.plot(w, c, 'ro')
    plt.show()


# 문제
# 텐서플로를 사용해서 cost 그래프를 그려보세요
def show_cost_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.placeholder(tf.float32)

    hx = w * x
    loss = tf.reduce_mean((hx - y) ** 2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(-30, 50):
        c = sess.run(loss, {w: i/10})

        plt.plot(i/10, c, 'ro')
    plt.show()

    sess.close()


def show_cost_3():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.placeholder(tf.float32)

    hx = w * x
    loss = tf.reduce_mean((hx - y) ** 2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ww, cc = [], []
    for i in range(-30, 50):
        c = sess.run(loss, {w: i/10})

        ww.append(i / 10)
        cc.append(c)

    sess.close()

    plt.plot(ww, cc, 'ro')
    plt.show()


def show_cost_4():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.placeholder(tf.float32)

    hx = w * x
    loss = tf.reduce_mean((hx - y) ** 2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # ww, cc = [], []
    # for i in np.arange(-3, 5, 0.1):
    #     c = sess.run(loss, {w: i})
    #
    #     ww.append(i)
    #     cc.append(c)

    ww = np.arange(-3, 5, 0.1)
    cc = []
    for i in ww:
        c = sess.run(loss, {w: i})
        cc.append(c)

    sess.close()

    plt.plot(ww, cc, 'ro')
    plt.show()


def show_cost_5():
    x = [1, 2, 3]
    y = [1, 2, 3]

    ww = np.arange(-3, 5, 0.1).reshape(-1, 1)
    cc = np.mean((ww * x - y) ** 2, axis=1)

    plt.plot(ww, cc, 'ro')
    plt.show()


# show_cost()
# show_cost_2()
# show_cost_3()
# show_cost_4()
show_cost_5()


