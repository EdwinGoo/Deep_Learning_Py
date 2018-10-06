# Day_03_04_SoftmaxRegression_iris.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection


def get_iris():
    df = pd.read_csv('Data/iris.csv')
    # print(df)

    iris = df.values
    # print(iris[:3])
    # print('-' * 50)

    xx = np.float32(df.values[:, :-1])
    xx = preprocessing.add_dummy_feature(xx)
    xx = np.float32(xx)
    # print(xx.shape, xx.dtype)
    # print(xx[:3])

    # yy = df.variety
    yy = df.variety.values
    # print(yy.shape, yy.dtype)
    # print(yy[:3])

    yy = preprocessing.LabelBinarizer().fit_transform(yy)
    # print(yy[:3])
    # print('-' * 50)

    # print(xx.shape, yy.shape)
    return xx, yy


def get_iris_sparse():
    df = pd.read_csv('Data/iris.csv')

    xx = np.float32(df.values[:, :-1])
    xx = preprocessing.add_dummy_feature(xx)
    xx = np.float32(xx)

    yy = df.variety.values

    yy = preprocessing.LabelEncoder().fit_transform(yy)
    print(yy[:3])
    print('-' * 50)

    print(xx.shape, yy.shape)
    return xx, yy


# 문제
# iris 데이터셋에 대해서
# train_set 70%로 학습하고 test_set 30%에 대해 정확도를 알려주세요.
def softmax_regression_iris_1():
    xx, y = get_iris()

    # 셔플 해야함.
    train_size = int(len(xx) * 0.7)
    x_train, x_test = xx[:train_size], xx[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3]))

    # (150, 3) = (150, 5) @ (5, 3)
    # (105, 3) = (105, 5) @ (5, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {x: x_train})
        print(i, sess.run(loss, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    print(pred.shape, y_test.shape)

    print(pred[:3])
    print(y_test[:3])
    print('-' * 50)

    pred_arg = np.argmax(pred, axis=1)
    test_arg = np.argmax(y_test, axis=1)

    print(pred_arg[:10])
    print(test_arg[:10])
    print('-' * 50)

    equals = (pred_arg == test_arg)
    print(equals[:10])
    print('acc :', np.mean(equals))

    sess.close()


def softmax_regression_iris_2():
    xx, y = get_iris()

    # 75:25
    # data = model_selection.train_test_split(xx, y)
    # 7:3
    # data = model_selection.train_test_split(xx, y, train_size=0.7)
    # 105:45
    data = model_selection.train_test_split(xx, y, train_size=105)
    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3]))

    # (105, 3) = (105, 5) @ (5, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {x: x_train})
        print(i, sess.run(loss, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    print(pred.shape, y_test.shape)

    print(pred[:3])
    print(y_test[:3])
    print('-' * 50)

    pred_arg = np.argmax(pred, axis=1)
    test_arg = np.argmax(y_test, axis=1)

    print(pred_arg[:10])
    print(test_arg[:10])
    print('-' * 50)

    equals = (pred_arg == test_arg)
    print(equals[:10])
    print('acc :', np.mean(equals))

    sess.close()


def softmax_regression_iris_3():
    xx, y = get_iris_sparse()

    # 75:25
    data = model_selection.train_test_split(xx, y)
    x_train, x_test, y_train, y_test = data

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3]))

    # (105,) = (105, 5) @ (5, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {x: x_train})
        print(i, sess.run(loss, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    print(pred.shape, y_test.shape)

    print(pred[:3])
    print(y_test[:3])
    print('-' * 50)

    pred_arg = np.argmax(pred, axis=1)

    print(pred_arg[:10])
    print('-' * 50)

    equals = (pred_arg == y_test)
    print(equals[:10])
    print('acc :', np.mean(equals))

    sess.close()


# softmax_regression_iris_1()
# softmax_regression_iris_2()
softmax_regression_iris_3()
