# softmax regression 드디어... 만났드아
import numpy as np
import tensorflow as tf

def softmax_my(array) :
    v = np.exp(array)
    print(v)
    print(v/np.sum(v))

softmax_my([2.0,1.0,0.1])

##