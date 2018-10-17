import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.image import imread

mnist = input_data.read_data_sets('mnist', one_hot=True)
x= mnist.train.images[0]
y= mnist.train.labels
# plt.imshow(x.reshape(28, 28), 'Greys')
# plt.show()

print(np.argmax(y,axis=1))