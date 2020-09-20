#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *

####################################_LSTM_##########################################

def modelLSTM(trainx,trainy):
    nb_classes = trainy.shape[1]

    n_timesteps = trainx.shape[1]
    n_features = trainx.shape[2]
    
    model = Sequential(name='LSTM')
    model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
    model.add(LSTM(64))
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model


def modelLSTMEmbedded(trainx,trainy,vocab_size):
    nb_classes = trainy.shape[1]
    input_length = trainx.shape[1]
    
    model = Sequential(name='LSTM_Embedded')
    model.add(Embedding(input_dim = vocab_size+1, output_dim = 64, input_length=input_length, mask_zero=True))
    model.add(LSTM(64))
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model

###################################_FCN_##########################################

def modelFCN(trainx,trainy):
    
    nb_classes = trainy.shape[1]
    
    s1 = trainx.shape[1]
    s2 = trainx.shape[2]
    
    input_layer = Input(shape=((s1,s2)))

    mask = Masking(mask_value=0.0)(input_layer)
    
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(mask)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv3 = Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    gap_layer = GlobalAveragePooling1D()(conv3)
    
    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name="FCN")
    
    return model


def modelFCNEmbedded(trainx,trainy,vocab_size):
    
    nb_classes = trainy.shape[1]
    

    n_timesteps = trainx.shape[1]
    

    input_layer = Input(shape=((n_timesteps,)))
    
    embedding = Embedding(input_dim = vocab_size+1, output_dim = 64, input_length=n_timesteps, mask_zero=True) (input_layer)

    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(embedding)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv3 = Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    gap_layer = GlobalAveragePooling1D()(conv3)
    
    x = Dropout(0.5)(gap_layer)

    output_layer = Dense(nb_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer, name="FCN_Embedded")
    
    return model