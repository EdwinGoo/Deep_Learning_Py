from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.misc import imresize 
from keras.preprocessing import image


mozzi = imread('data/santafe.jpg')
mozzi = imresize(mozzi, (32,32,3))
mozziNew = np.array([mozzi])
mozzi = mozzi.astype('float32')/255. #모찌는 640 640 3 , 우리 모델은 32,32,3

vgg16 = VGG16(weights=None)
vgg16.summary()
vgg16.load_weights('DAY_08/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

