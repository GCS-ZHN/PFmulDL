#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import numpy as np
import pandas as pd
import math

AAINDEX = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}

MAXLEN = 2000
def to_onehot(seq, start=0):
    onehot = np.zeros((MAXLEN, 21), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot

class DFGenerator(Sequence):
    def __init__(self, df, terms_dict, nb_classes, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.terms_dict = terms_dict
    def __len__(self):                                                                                                                   
        return np.ceil(len(self.df) / float(self.batch_size)).astype(np.int32)   
    def __getitem__(self, idx):                                                                                                          
        batch_index = np.arange(idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))                                                          
        df = self.df.iloc[batch_index]                                                                                                   
        data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            onehot = to_onehot(seq)
            data_onehot[i, :, :] = onehot
            for t_id in row.prop_annotations:
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1
        self.start += self.batch_size
        return (data_onehot, labels)
    def __next__(self):
        return self.next()
    def reset(self):
        self.start = 0
    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.int32)
            labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                seq = row.sequences
                onehot = to_onehot(seq)
                data_onehot[i, :, :] = onehot
                for t_id in row.prop_annotations:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data_onehot, labels)
        else:
            self.reset()
            return self.next()


# In[6]:


def plot_curve(history):
    plt.figure()
    x_range = range(0,len(history.history['loss']))
    plt.plot(x_range, history.history['loss'],'bo',label='Training loss')
    plt.plot(x_range, history.history['val_loss'],'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def train_model(model_path, data_path, valid_path='none', epochs=5, batch_size=32, data_size=100000, terms_file='data/terms.pkl'):
    model = load_model(model_path)
#     model.summary()
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    tbCallBack = TensorBoard(log_dir="./model", histogram_freq=1,write_grads=True)
    logger = CSVLogger('result/train_log.txt')
    with open(terms_file,'rb') as file:
        terms_df = pd.read_pickle(file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    with open(data_path, 'rb') as file:
        data_df = pd.read_pickle(file)
    if len(data_df)> data_size:
        data_df = data_df.sample(n=data_size)

    if valid_path =='none':
        valid_df = data_df.sample(frac=0.2)
        train_df = data_df[~data_df.index.isin(valid_df.index)]
    else:
        train_df = data_df
        with open(valid_path, 'rb') as file:
            valid_df = pd.read_pickle(file)
    valid_steps = int(math.ceil(len(valid_df) / batch_size))
    train_steps = int(math.ceil(len(train_df) / batch_size))
    train_generator = DFGenerator(train_df, terms_dict, nb_classes, batch_size)
    valid_generator = DFGenerator(valid_df, terms_dict, nb_classes, batch_size)
    #     训练模型
    his =  model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        max_queue_size=batch_size,
        workers=12,
        callbacks=[tbCallBack, logger, checkpointer, earlystopper])
    plot_curve(his)






