from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('mnist', one_hot=True, reshape=False)
X_train= mnist.train.images

print(X_train.shape)