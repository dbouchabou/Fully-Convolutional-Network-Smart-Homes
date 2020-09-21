#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *

####################################_LSTM_##########################################

def model(trainx,trainy):
    nb_classes = trainy.shape[1]

    n_timesteps = trainx.shape[1]
    n_features = trainx.shape[2]
    
    model = Sequential(name='LSTM')
    model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
    model.add(LSTM(64))
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model