# Day_03_02_SoftmaxRegression.py
import numpy as np
import tensorflow as tf


def softmax(array):
    v = np.exp(array)
    print(v)
    print(v / np.sum(v))


def softmax_regression_1():
    x = [[1., 1., 1.],      # C
         [1., 1., 2.],
         [1., 3., 2.],      # B
         [1., 2., 4.],
         [1., 5., 3.],      # A
         [1., 4., 4.]]
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],         # [0.1, 0.6, 0.3]
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    # w = tf.Variable(tf.random_uniform([3, 3]))
    w = tf.Variable(tf.zeros([3, 3]))

    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# 3시간 공부하고 4번 출석한 학생과
# 6시간 공부하고 2번 출석한 학생의 학점을 알려주세요
def softmax_regression_2():
    xx = [[1., 1., 1.],      # C
          [1., 1., 2.],
          [1., 3., 2.],      # B
          [1., 2., 4.],
          [1., 5., 3.],      # A
          [1., 4., 4.]]
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([3, 3]))

    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(loss, {x: xx}))
    print('-' * 50)

    pred = sess.run(hx, {x: [[1., 3., 4.],
                             [1., 6., 2.]]})
    print(pred)

    pred_arg = np.argmax(pred, axis=1)
    print(pred_arg)

    grades = np.array(['A', 'B', 'C'])
    print(grades[pred_arg[0]], grades[pred_arg[1]])
    print(grades[pred_arg])

    sess.close()


# 문제
# 행렬 곱셈에서 w를 앞쪽에 두세요
def softmax_regression_3():
    x = [[1., 1., 1.],      # C
         [1., 1., 2.],
         [1., 3., 2.],      # B
         [1., 2., 4.],
         [1., 5., 3.],      # A
         [1., 4., 4.]]
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]
    x = np.float32(np.transpose(x))
    y = np.transpose(y)

    w = tf.Variable(tf.zeros([3, 3]))

    # (3, 6) = (3, 3) @ (3, 6)
    z = tf.matmul(w, x)

    # (3, 6)
    hx = tf.nn.softmax(z, axis=0)

    # (6,)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z, dim=0)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('z :', z.shape)
    print(sess.run(z))
    print('-' * 50)

    print('hx :', hx.shape)
    print(sess.run(hx))
    print('-' * 50)

    print('loss_i :', loss_i.shape)
    print(sess.run(loss_i))
    print('-' * 50)

    print('loss :', loss.shape)
    print(sess.run(loss))

    # for i in range(10):
    #     sess.run(train)
    #     print(i, sess.run(loss))

    sess.close()


# softmax([2.0, 1.0, 0.1])

# softmax_regression_1()
# softmax_regression_2()
softmax_regression_3()












