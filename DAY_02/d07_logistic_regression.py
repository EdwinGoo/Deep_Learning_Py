import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def show_sigmoid() :
    def sigmoid(z) :
        return 1 / (1+np.e ** -z)

    for z in np.linspace(-5, 5, 30) :
        s = sigmoid(z)
        plt.plot(z,s, "bo")

    plt.show()

def select_log() :
    def log_a() :
        return 'A'
    def log_b() :
        return 'B'
    
    print(y * log_a() + (1-y) * log_b())

def logistic_regression() :
    x = [[1.,1.,1.,1.,1.,1.],[1.,1.,2.,3.,4.,6.], [1.,3.,2.,5.,4.,2.]] 
    y = [[0,0,0,1,1,1]]
    y = np.int32(y)

    w = tf.Variable(tf.random_uniform([1,3]))
    z =  tf.matmul(w ,x)
    hx = 1  / (1 + tf.exp(-z))

    loss_i = y * -tf.log(hx) + ([1]-y) * -tf.log(1-hx) 

    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        sess.run(train)
        # if i%200 == 0 :
            # print(i,sess.run(loss))
    print(i,sess.run(hx))
    
    sess.close()

logistic_regression()