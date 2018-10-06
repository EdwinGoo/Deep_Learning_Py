import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing, model_selection 

def get_iris():
    #dataframe -> df 
    df = pd.read_csv("./DATA/iris.csv")
    # iris = df.values print(iris) 
    xx = np.float32(preprocessing.add_dummy_feature(df.values[:, :-1]))
    yy =df.variety
    yy = preprocessing.LabelBinarizer().fit_transform(yy)
    
    #애초에 전처리에서 셔플하고 나오는게 편할 듯

    print(xx.shape)
    print(yy.shape)
    return xx, yy

def get_iris_sparse():
    df = pd.read_csv("./DATA/iris.csv")
    xx = np.float32(preprocessing.add_dummy_feature(df.values[:, :-1]))
    yy =df.variety
    yy = preprocessing.LabelEncoder().fit_transform(yy)
    
    # labelEncoder
    # yy = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    # 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
    # 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    # 2 2]
    return xx, yy

def softmax_regression_iris_미완성_데이터정렬이라서() :
    xx, y = get_iris()

    train_size = int(len(xx)*0.7)
    x_train, x_test= xx[:train_size] , xx[train_size:]
    y_train, y_test= y[:train_size] , y[train_size:]

    # xx = [[1.,1.,1.],
    #       [1.,1.,2.],
    #       [1.,3.,2.],
    #       [1.,2.,4.],
    #       [1.,5.,3.],
    #       [1.,4.,4.]] # 6 x 3
    # y = [[0,0,1],
    #      [0,0,1],
    #      [0,1,0],
    #      [0,1,0],
    #      [1,0,0],
    #      [1,0,0]]
    ## xx 150 * 5
    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3]))

    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z) 
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train, {x:x_train})

    pred = sess.run(hx, {x:x_test})
    print(pred)

    sess.close()

# softmax_regression_iris_미완성_데이터정렬이라서()

def softmax_regression_iris2() :
    xx, y = get_iris()

    # data = model_selection.train_test_split(xx,y)
    data = model_selection.train_test_split(xx,y, train_size=0.7)
    x_train, x_test, y_train, y_test= data

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3]))

    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z) 
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        sess.run(train, {x:x_train})
        if i%2500 == 0 :
            print(i,sess.run(loss, {x:x_train}))

    pred = sess.run(hx, {x:x_test})
    print(pred.shape, y_test.shape)

    print('-' * 50)
    print(pred[:3])
    print(y_test[:3])
    print('-' * 50)

    pred_arg = np.argmax(pred, axis=1)
    test_arg = np.argmax(y_test, axis=1)

    print(pred_arg[:10])
    print(test_arg[:10])
    print('-' * 50)

    equals = (pred_arg == test_arg)
    print(equals[:10])
    print('acc :', np.mean(equals))
    sess.close()

# softmax_regression_iris2()

def softmax_regression_iris3() :
    xx, y = get_iris_sparse()

    data = model_selection.train_test_split(xx,y, train_size=0.7)

    x_train, x_test, y_train, y_test= data
 
    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3]))

    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z) 
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x:x_train})
        if i%250 == 0 :
            print(i,sess.run(loss, {x:x_train}))

    pred = sess.run(hx, {x:x_test})
    print(pred.shape, y_test.shape)

    print('-' * 50)
    print(pred[:3])
    print(y_test[:3])
    print('-' * 50)

    pred_arg = np.argmax(pred, axis=1)

    print(pred_arg[:10])
    print(y_test[:10])
    print('-' * 50)

    equals = (pred_arg == y_test)
    print(equals[:10])
    print('acc :', np.mean(equals))
    sess.close()

# softmax_regression_iris3()