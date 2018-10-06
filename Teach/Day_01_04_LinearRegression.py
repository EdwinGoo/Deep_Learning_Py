# Day_01_04_LinearRegression.py
import tensorflow as tf


def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.)
    b = tf.Variable(10.)

    hx = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_mean(tf.square(hx - y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_mean(tf.square(tf.add(tf.multiply(w, x), b) - y)))
    # print('dinner', print('hello'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # print(w)
    # print(sess.run(w))

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print('5 :', w * 5)

    ww = sess.run(w)
    bb = sess.run(b)

    print('5 :', ww * 5 + bb)
    print('7 :', ww * 7 + bb)

    sess.close()

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     for i in range(10):
    #         sess.run(train)


def linear_regression_2():
    xx = [1, 2, 3]
    yy = [1, 2, 3]

    w = tf.Variable(10.)
    b = tf.Variable(10.)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    hx = w * x + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={x: xx, y: yy})
        print(i, sess.run(loss, {x: xx, y: yy}))
    print('-' * 50)

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print(sess.run(hx, {x: 5}))
    print(sess.run(hx, {x: 7}))
    print(sess.run(hx, {x: xx}))
    print(sess.run(hx, {x: [1, 2, 3]}))
    print(sess.run(hx, {x: [5, 7]}))

    sess.close()


def linear_regression_3():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.)
    b = tf.Variable(10.)

    x = tf.placeholder(tf.float32)

    hx = w * x + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={x: xx})
        print(i, sess.run(loss, {x: xx}))
    print('-' * 50)

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print(sess.run(hx, {x: 5}))
    print(sess.run(hx, {x: 7}))
    print(sess.run(hx, {x: xx}))
    print(sess.run(hx, {x: [1, 2, 3]}))
    print(sess.run(hx, {x: [5, 7]}))

    sess.close()


# linear_regression_1()
# linear_regression_2()
linear_regression_3()
