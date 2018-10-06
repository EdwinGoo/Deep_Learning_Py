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

# logistic_regression()


def logistic_regression2() :
    x_data = [[1.,1.,1.,1.,1.,1.],[1.,1.,2.,3.,4.,6.], [1.,3.,2.,5.,4.,2.]] 
    y_data = [[0,0,0,1,1,1]]
    # y = np.int32(y)
    y = np.float32(y_data)

    w = tf.Variable(tf.random_uniform([1,3]))
    # w = tf.Variable(tf.zeros([1,3])) test
    
    x = tf.placeholder(tf.float32) 

    z =  tf.matmul(w ,x)
    # hx = 1  / (1 + tf.exp(-z)) 아래 함수와 같다.
    hx = tf.nn.sigmoid(z)
    # loss_i = y * -tf.log(hx) + ([1]-y) * -tf.log(1-hx)  아래 함수와 같다.
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z) 

    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        sess.run(train, {x:x_data})
        if i%2500 == 0 :
            print(i,sess.run(loss, {x:x_data}))

    print('■' * 100)
    print(sess.run(hx, {x:x_data}))
    print('■' * 100)
    print(sess.run(hx, {x: [[1], [3], [4]]})>0.5)
    print(sess.run(hx, {x: [[1], [5], [2]]})>0.5)
    print(sess.run(hx, {x: [[1,1], [1,3], [2,8]]})>0.5)
    print('■' * 100)

    sess.close()

logistic_regression2()



def logistic_regression_not_sigmoid() :
    x_data = [[1.,1.,1.,1.,1.,1.],[1.,1.,2.,3.,4.,6.], [1.,3.,2.,5.,4.,2.]] 
    y_data = [[0,0,0,1,1,1]]
    # y = np.int32(y)
    y = np.float32(y_data)

    w = tf.Variable(tf.random_uniform([1,3]))
    # w = tf.Variable(tf.zeros([1,3])) test
    
    x = tf.placeholder(tf.float32) 

    z =  tf.matmul(w ,x)
    # hx = 1  / (1 + tf.exp(-z)) 아래 함수와 같다.
    hx = tf.nn.sigmoid(z)
    # loss_i = y * -tf.log(hx) + ([1]-y) * -tf.log(1-hx)  아래 함수와 같다.
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z) 

    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        sess.run(train, {x:x_data})
        if i%2500 == 0 :
            print(i,sess.run(loss, {x:x_data}))

    print('■' * 100)
    print(sess.run(z, {x: [[1], [3], [4]]})>0) 
    # sigmoid를 사용하지 않다면 0.5보다 크려면 1/(1+1) e의 0승 다중레이어에서 값을 유지할 때 사용
    print('■' * 100)

    sess.close()

logistic_regression_not_sigmoid()