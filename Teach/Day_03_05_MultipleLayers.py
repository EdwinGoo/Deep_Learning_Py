# Day_03_05_MultipleLayers.py
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def mnist_basic():
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    print(mnist.train.images.shape)         # (55000, 784)
    print(mnist.validation.images.shape)    # (5000, 784)
    print(mnist.test.images.shape)          # (10000, 784)

    print(mnist.train.labels.shape)         # (55000, 10)
    print(mnist.train.labels[:5])

    print(mnist.train.num_examples)


def show_accuracy(sess, hx, x, y, title, dataset):
    pred_arg = tf.argmax(hx, axis=1)
    test_arg = tf.argmax(y, axis=1)

    equals = tf.equal(pred_arg, test_arg)
    equals_float = tf.cast(equals, tf.float32)
    mean = tf.reduce_mean(equals_float)
    print(title, sess.run(mean, {x: dataset.images,
                                 y: dataset.labels}))


def mnist_softmax(x):
    w = tf.Variable(tf.random_uniform([784, 10]))
    b = tf.Variable(tf.random_uniform([10]))

    # (55000, 10) = (55000, 784) @ (784, 10)
    return tf.matmul(x, w) + b
    # return tf.nn.softmax(z)


def mnist_multi_layers(x):
    w1 = tf.Variable(tf.truncated_normal([784, 256]))
    w2 = tf.Variable(tf.truncated_normal([256, 256]))
    w3 = tf.Variable(tf.truncated_normal([256, 10]))

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    return tf.matmul(r2, w3) + b3


def mnist_multi_layers_xavier(x):
    w1 = tf.get_variable('w1', shape=[784, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2', shape=[256, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3', shape=[256, 10],
                         initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    return tf.matmul(r2, w3) + b3


def show_model(model):
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # ------------------------ #

    z = model(x)

    # ------------------------ #

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)        # RMSProp
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 15
    batch_size = 100
    iteration = mnist.train.num_examples // batch_size

    for i in range(epochs):
        total = 0
        for j in range(iteration):
            xx, yy = mnist.train.next_batch(batch_size)

            _, c = sess.run([train, loss], {x: xx, y: yy})
            total += c

        print(i, total / iteration)
    print('-' * 50)

    show_accuracy(sess, z, x, y, 'train :', mnist.train)
    show_accuracy(sess, z, x, y, 'valid :', mnist.validation)
    show_accuracy(sess, z, x, y, 'test  :', mnist.test)

    sess.close()



# mnist_basic()

# show_model(mnist_softmax)
# show_model(mnist_multi_layers)
show_model(mnist_multi_layers_xavier)


# [1] mnist_softmax
# 14 0.2889897256005894
# --------------------------------------------------
# train : 0.9206727
# valid : 0.9228
# test  : 0.9174

# [2] mnist_multi_layers
# 14 0.4017462790712582
# --------------------------------------------------
# train : 0.9823273
# valid : 0.9642
# test  : 0.9605

# [3] mnist_multi_layers_xavier
# 14 0.06895155501059806
# --------------------------------------------------
# train : 0.9878
# valid : 0.9732
# test  : 0.9728

print('\n\n\n\n\n\n\n')