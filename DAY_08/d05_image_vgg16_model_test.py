from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.misc import imresize 
from keras.preprocessing import image

mozzi = image.load_img('data/mozzi.jpg', target_size=(224,224))
x = image.img_to_array(mozzi)
x = preprocess_input(x)

bullmastiff = image.load_img('data/bullmastiff.jpg', target_size=(224,224))
x2 = image.img_to_array(bullmastiff)
x2 = preprocess_input(x2)

xNew = np.array([x,x2])

#vgg16 모델의 가중치 읽기
vgg16 = VGG16(weights=None)
vgg16.summary()
vgg16.load_weights('DAY_08/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

Y_pred = vgg16.predict(xNew)
print(np.argmax(Y_pred)) # 153
print(np.max(Y_pred, axis=1))