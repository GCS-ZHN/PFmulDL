#!/usr/bin/env python
# coding: utf-8

# In[2]:


#要求的库与参数
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import(
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate, TimeDistributed,
    MaxPooling1D, Dropout, RepeatVector, Layer, Reshape, SimpleRNN, LSTM, BatchNormalization
)
import tensorflow as tf
import numpy as np
import pandas as pd
import math

AAINDEX = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
MAXLEN = 2000

def pre_model(model_file, term_file='data/terms.pkl'):
    MAXLEN = 2000
    with open(term_file, 'rb') as file:
        terms_df = pd.read_pickle(file)
    terms = terms_df['terms'].values.flatten()
    batch_size = 32
    params = {
        'max_kernel': 65,
        'initializer': 'glorot_normal',
        'dense_depth': 0,
        'nb_filters': 256,
        'optimizer': Adam(lr=2e-4),
        'loss': 'binary_crossentropy'
    }
    nb_classes = len(terms)
    inp_hot = Input(shape=(MAXLEN, 21), dtype=np.float32)
    kernels = range(8, params['max_kernel'], 8)

    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=kernels[i],
            padding='same',
            name='conv_' + str(i),
            kernel_initializer=params['initializer'])(inp_hot)
        batch_norm = BatchNormalization()(conv)
        relu_ = tf.keras.activations.relu(batch_norm)
        nets.append(relu_)
    new_nets = []
    for i in range(len(nets)):
        new_conv = Conv1D(filters=params['nb_filters'],
                          kernel_size=kernels[i],
                          padding='valid',
                          name='new_conv_' + str(i),
                          kernel_initializer=params['initializer'])(nets[i])
        new_pool = MaxPooling1D(MAXLEN - kernels[i] + 1, name=str(i) + '_pool')(new_conv)
        flat = Flatten(name='flat_' + str(i))(new_pool)
        new_nets.append(flat)

    net = Concatenate(axis=1)(new_nets)
    # net = tf.keras.layers.Add()([pool_i, net])
    # net = Flatten()(net)
    net = Dense(nb_classes, activation='sigmoid')(net)
    model = Model(inputs=inp_hot, outputs=net)
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    model.save(model_file)
    model.summary()
    # print(model)
    
def crnn_model(model_file, load=False, origin_path='origin_model.h5',
               term_file='data/terms.pkl'):
    MAXLEN = 2000
    with open(term_file, 'rb') as file:
        terms_df = pd.read_pickle(file)
    terms = terms_df['terms'].values.flatten()
    batch_size = 32
    params = {
        'max_kernel': 65,
        'initializer': 'glorot_normal',
        'dense_depth': 0,
        'nb_filters': 256,
        'optimizer': Adam(lr=0.0005),
        'loss': 'binary_crossentropy'
    }
    nb_classes = len(terms)
    inp_hot = Input(shape=(MAXLEN, 21), dtype=np.float32)
    kernels = range(8, params['max_kernel'], 8)

    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=kernels[i],
            padding='same',
            name='conv_' + str(i),trainable=False,
            kernel_initializer=params['initializer'])(inp_hot)
        batch_norm = BatchNormalization()(conv)
        relu_ = tf.keras.activations.relu(batch_norm)
        nets.append(relu_)
    new_nets = []
    for i in range(len(nets)):
        new_conv = Conv1D(filters=params['nb_filters'],
                          kernel_size=kernels[i],
                          padding='valid',
                          name='new_conv_' + str(i),trainable=False,
                          kernel_initializer=params['initializer'])(nets[i])
        new_pool = MaxPooling1D(MAXLEN - kernels[i] + 1, name=str(i) + '_pool')(new_conv)
        flat = Flatten(name='flat_' + str(i))(new_pool)
        new_nets.append(flat)

    net = Concatenate(axis=1)(new_nets)
    net = BatchNormalization()(net)
    net = Dropout(0.5)(net)
    cnn_out = Dense(512, activation='relu')(net)
    net = Dropout(0.5)(cnn_out)
    net = RepeatVector(11)(net)
    net = GRU(256, activation='tanh', return_sequences=True)(net)
    net = GRU(256, activation='tanh', return_sequences=True)(net)
    net = GRU(256, activation='tanh', return_sequences=True)(net)
    net = Flatten()(net)
    net = Dense(nb_classes, activation='sigmoid')(net)
    model = Model(inputs=inp_hot, outputs=net)
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    if load:
        loaded_model = load_model(origin_path)
        old_weights = loaded_model.get_weights()
        now_weights = model.get_weights()
        cnt = 0
        for i in range(len(now_weights)):
            if old_weights[cnt].shape == now_weights[i].shape:
                now_weights[i] = old_weights[cnt]
                cnt = cnt + 1
        print(f'{cnt} layers weights copied, total {len(now_weights)}')
        model.set_weights(now_weights)
    model.save(model_file)
    model.summary()




