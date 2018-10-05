#선형회귀
import tensorflow as tf

def random_uniform_test() :
    uni_rand = tf.Variable(tf.random_uniform([15]))
    uni_rand_matrix = tf.Variable(tf.random_uniform([3,5]))
    normal_rand = tf.Variable(tf.random_normal([15]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(sess.run(uni_rand))
    print(sess.run(normal_rand))
    print(sess.run(uni_rand_matrix))

    sess.close()

# random_uniform_test()

def multiple_regression0() :
    x = [2,1,3]
    y = [2,1,3]

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w * x + b
    loss = tf.reduce_mean((hx-y)**2)

    optimizer = tf.train.GradientDescentOptimizer(0.15)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        sess.run(train)
        print(i,sess.run(loss))
    sess.close()

# multiple_regression()

def multiple_regression1() :
    x = [[1,0,3,0,5],
         [0,2,0,4,0]]
    y = [1,2,3,4,5]

    w = tf.Variable(tf.random_uniform([2]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w[0] * x[0]  + w[1] * x[1] + b
    
    loss = tf.reduce_mean((hx-y)**2)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        if i%200 == 0 :
            print(i,sess.run(loss))
    sess.close()

# multiple_regression1()


def multiple_regression2_del_basis() :
    x = [[1.,1.,1.,1.,1.],[1,0,3,0,5], [0,2,0,4,0]] # 3 X 5
    y = [1,2,3,4,5]

    # 1 X 5  x 행렬에 1 배열을 추가함으로써 b를 계산하는 것과 같은 효과를 줌 b[] = w[2] * x[2]

    w = tf.Variable(tf.random_uniform([1,3])) # 1 X 3
    hx =  tf.matmul(w , x)
    
    loss = tf.reduce_mean((hx-y)**2)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1001):
        sess.run(train)
        if i%200 == 0 :
            print(i,sess.run(loss))
    print(sess.run(w))
    sess.close()

multiple_regression2()