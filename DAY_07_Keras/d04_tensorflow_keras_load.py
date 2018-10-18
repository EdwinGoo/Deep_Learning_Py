import keras
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# model = load_model('my_keras_relu_he_adam_epchs100_batch_size100_V20.h5')
# model = load_model('my_keras_sig_norm_sgd_epchs100_batch_size100_V20.h5')

historyDFrelu = pd.read_pickle("./my_keras_relu_he_adam_epchs100_batch_size100_V20.pkl")
historyDFsig = pd.read_pickle("./add_layer.pkl")

historyDFrelu[['loss', 'val_loss']].plot(style={'loss':'g-', 'val_loss': 'r-'})
historyDFsig[['loss', 'val_loss']].plot(style={'loss':'g-', 'val_loss': 'r-'})

historyDFrelu[['acc', 'val_acc']].plot(style={'loss':'g-', 'val_loss': 'r-'})
historyDFsig[['acc', 'val_acc']].plot(style={'loss':'g-', 'val_loss': 'r-'})

plt.show()

