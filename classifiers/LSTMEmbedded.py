#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *

####################################_LSTM_Embedded_##########################################

def model(trainx,trainy,vocab_size):
    nb_classes = trainy.shape[1]
    input_length = trainx.shape[1]
    
    model = Sequential(name='LSTM_Embedded')
    model.add(Embedding(input_dim = vocab_size+1, output_dim = 64, input_length=input_length, mask_zero=True))
    model.add(LSTM(64))
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model