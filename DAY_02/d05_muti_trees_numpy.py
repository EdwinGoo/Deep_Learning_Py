#선형회귀
import tensorflow as tf
import numpy as np

def multiple_regression_tree_placeholder() :
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
    # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32).T
    trees = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32)

    xx = np.float32([trees[:,0],trees[:,1]])
    y = trees[:,2]

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


# def fanxy_index_get_data() :
#     # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', unpack=True, dtype=np.float32)
#     # girth, height, volume = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32).T
#     trees = np.loadtxt('./data/trees.csv', delimiter=',', dtype=np.float32)

#     print(trees[:,1])
#     print(trees[:,0])

# fanxy_index_get_data()



