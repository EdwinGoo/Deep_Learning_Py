import tensorflow as tf
import numpy as np

def multiple_regression_tree_fanxy_index() :
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32).T
    
    ########################## 전처리 과정에서 Transpose는 성능상 문제가 되지 않음, 다만 반복문에서 사용할 경우 주의 ######################################
    trees = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32)

    xx = np.float32([trees[:,0],trees[:,1]])
    print(xx.shape)
    y = trees[:,2]
    ########################## 전처리 과정에서 Transpose는 성능상 문제가 되지 않음, 다만 반복문에서 사용할 경우 주의 ######################################

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

# multiple_regression_tree_fanxy_index()


def multiple_regression_tree_fanxy_index2() :
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32).T
    
    ########################## 전처리 과정에서 Transpose는 성능상 문제가 되지 않음, 다만 반복문에서 사용할 경우 주의 ######################################
    trees = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32)
    xx = trees[:,:-1]
    print(xx.shape)
    y = trees[:, -1:]
    print(y.shape)

    ########################## 전처리 과정에서 Transpose는 성능상 문제가 되지 않음, 다만 반복문에서 사용할 경우 주의 ######################################

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([2, 1]))
    b = tf.Variable(tf.random_uniform([1]))
    
    hx =  tf.matmul(x,w) + b
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

# multiple_regression_tree_fanxy_index2()


def fanxy_index_get_data() :
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32).T
    trees = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32)
    print(trees)
    print(trees[:,1])  
    print(trees[:,0])

# fanxy_index_get_data()



