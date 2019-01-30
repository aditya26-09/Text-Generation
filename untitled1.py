# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:04:31 2019

@author: Aditya
"""

#LSTM network to generate text for "Alice in the Wonderland"
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#opening the text
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

#creating mapping 
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

#analyzing number of characters and words
n_chars = len(chars)
n_vocab = len(raw_text)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0,n_vocab-seq_length,1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

#preparing the training set for LSTM
X = np.reshape(dataX,(n_patterns,seq_length,1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#fitting the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)