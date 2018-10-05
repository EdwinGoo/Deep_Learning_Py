#선형회귀
import tensorflow as tf
import numpy as np
#"Girth","Height","Volume"


def multiple_regression_tree() :

    girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    x = np.float32([girth,height])
    y = volume

    w = tf.Variable(tf.random_uniform([1,2]))
    b = tf.Variable(tf.random_uniform([1]))

    hx =  tf.matmul(w,x)+b

    loss = tf.reduce_mean((hx-y)**2)


    optimizer = tf.train.GradientDescentOptimizer(0.00015)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(201):
        sess.run(train)
        if i%20 == 0 :
            print(i,sess.run(loss))
    print(sess.run(w))
    sess.close()

# # multiple_regression_tree()

def multiple_regression_tree_placeholder() :


    girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    # xx = [[1] * 31, girth, height]
    xx = np.float32([girth,height])
    y = volume

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([1, 2]))
    b = tf.Variable(tf.random_uniform([1]))
    
    hx =  tf.matmul(w,x) + b
    loss = tf.reduce_mean((hx-y)**2)


    optimizer = tf.train.GradientDescentOptimizer(0.00015)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(201):
        sess.run(train, {x : xx})
        if i%20 == 0 :
            print(i,sess.run(loss, {x: xx}))
    print(sess.run(w))
    sess.close()

multiple_regression_tree_placeholder()

# def multiple_trees_2():
#     girth, height, volume = np.loadtxt('Data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)

#     xx = [[1] * 31, girth, height]
#     y = volume

#     x = tf.placeholder(tf.float32)
#     w = tf.Variable(tf.random_uniform([1, 3]))
    
#     # (1, 31) = (1, 3) @ (3, 31)
#     hx = tf.matmul(w, x)
#     loss = tf.reduce_mean((hx - y) ** 2)

#     optimizer = tf.train.GradientDescentOptimizer(0.0001)
#     train = optimizer.minimize(loss)

#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     for i in range(10):
#         sess.run(train, {x: xx})
#         print(i, sess.run(loss, {x: xx}))
#     print('-' * 50)
#     sess.close()


# multiple_trees_2()
