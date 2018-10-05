#선형회귀
import tensorflow as tf

def linear_regression() :
    x = [2,1,3]
    y = [2,1,3]

    w = tf.Variable(10.0) # 타입에러 있을 수 있다 가중치랑 바이어스는 실수이므로 10이 아니라 10.0
    b = tf.Variable(10.)

    hx = tf.add(tf.multiply(w,x),b)
    loss = tf.reduce_mean(tf.square(hx-y))

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.15)
    # train = optimizer.minimize(loss=loss)

    optimizer = tf.train.GradientDescentOptimizer(0.15)
    train = optimizer.minimize(loss)

    ## TENSORFLOW는 세션을 열어야한당
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        sess.run(train)
        # print(i, sess.run(loss))

    print(sess.run(w) * 5 + sess.run(b))
    print(sess.run(w) * 7 + sess.run(b))

    sess.close()

    # with  tf.Session() as sess :
    #     sess.run(tf.global_variables_initializer())
    #     for i in range(10):
    #         print(sess.run(train))

def linear_regression_2() :
    xx = [2,1,3]
    yy = [2,1,3]
    
    w = tf.Variable(10.0) # 타입에러 있을 수 있다 가중치랑 바이어스는 실수이므로 10이 아니라 10.0
    b = tf.Variable(10.)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    hx = w * x + b
    loss = tf.reduce_mean((hx-y)**2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        sess.run(train, feed_dict={x:xx, y:yy})
        # print(i,sess.run(loss, {x:xx, y:yy}))

    print(sess.run(hx,{x:5}))
    print(sess.run(hx,{x:xx}))
    print(sess.run(hx,{x:[2,1,3]}))
   
    sess.close()

def linear_regression_3() :

    xx = [2,1,3]
    y = [2,1,3]
    
    w = tf.Variable(10.0) # 타입에러 있을 수 있다 가중치랑 바이어스는 실수이므로 10이 아니라 10.0
    b = tf.Variable(10.)

    x = tf.placeholder(tf.float32)

    hx = w * x + b
    loss = tf.reduce_mean((hx-y)**2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        sess.run(train, feed_dict={x:xx})

    print(sess.run(hx,{x:5}))

    sess.close()

linear_regression_3()
