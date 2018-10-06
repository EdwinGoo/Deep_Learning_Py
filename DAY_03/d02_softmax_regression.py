# softmax regression 드디어... 만났드아
import numpy as np
import tensorflow as tf

def softmax_my(array) :
    v = np.exp(array)
    print(v)
    print(v/np.sum(v))

# softmax_my([2.0,1.0,0.1])

def softmax_regression() :
    x_data = [[1.,1.,1.],
              [1.,1.,2.],
              [1.,3.,2.],
              [1.,2.,4.],
              [1.,5.,3.],
              [1.,4.,4.]] # 6 x 3

    y_data = [[0,0,1],
              [0,0,1],
              [0,1,0],
              [0,1,0],
              [1,0,0],
              [1,0,0]]

    w = tf.Variable(tf.random_uniform([3,3]))
    
    z =  tf.matmul(x_data ,w) # y^ 추정
    hx = tf.nn.softmax(z) #분류에 대한 가설 여기서는 확률
    loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=z) 
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        sess.run(train)
        if i%2500 == 0 :
            print(i,sess.run(loss))

    sess.close()

# softmax_regression()

def softmax_regression2() :
    xx = [[1.,1.,1.],
          [1.,1.,2.],
          [1.,3.,2.],
          [1.,2.,4.],
          [1.,5.,3.],
          [1.,4.,4.]] # 6 x 3
    y = [[0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([3, 3]))

    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z) 
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x:xx})
        # print(i, sess.run(loss, {x: xx}))

    pred = sess.run(hx, {x: [[1., 5., 5.],
                             [1., 6., 5.]]})
    print(pred)

    sess.close()
    
# softmax_regression2()

def softmax_regression_transpose() :
    x_data = [[1.,1.,1.],
              [1.,1.,2.],
              [1.,3.,2.],
              [1.,2.,4.],
              [1.,5.,3.],
              [1.,4.,4.]] # 6 x 3
    
    x_data = np.transpose(x_data) # 3 X 6
    x_data = np.float32(x_data)

    y_data = [[0,0,1],
              [0,0,1],
              [0,1,0],
              [0,1,0],
              [1,0,0],
              [1,0,0]] # 6 3 
    y_data = np.transpose(y_data) # 3 X 6

    w = tf.Variable(tf.random_uniform([3,3]))
    
    z =  tf.matmul(w ,x_data) # 3 3 3 6 = 3 6 z 추정
    # z =  tf.matmul(x_data, w) # 6 3 3 3 = 6 3

    hx = tf.nn.softmax(z, axis=0) # 3, 6 분류에 대한 가설 여기서는 확률

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_data, logits=z, dim=0)  #버전2에서만 dim(axis)을 설정할 수 있다. 
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i,sess.run(loss))

    sess.close()

softmax_regression_transpose()
