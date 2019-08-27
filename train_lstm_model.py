'''
Train an LSTM model on the IMDB sentiment classification task.
'''
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM
from keras.datasets import imdb
from glove_utils import load_embedding
import pickle
import time
import tensorflow as tf

from data_utils import IMDBDataset

maxlen = 200
batch_size = 32

print('Loading data...')

(train_text, x_train, y_train), (test_text, x_test, y_test) = IMDBDataset.load_data()

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding = 'pre', truncating = 'pre')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding = 'pre', truncating = 'pre')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

GLOVE_FILENAME = 'data/glove.6B.100d.txt'
start_time = time.time()
word2index, index2word, index2embedding = load_embedding(GLOVE_FILENAME)
end_time = time.time() 
print('Loaded %s word vectors in %f seconds' % (len(word2index), end_time- start_time))

# dimensionality of word embeddings
EMBEDDING_DIM = 100



# Build and train model

print('Build model...')
model = Sequential()
model.add(Embedding(len(index2embedding), EMBEDDING_DIM, weights=[index2embedding], trainable=False, name= 'embedding'))
model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2, name = 'lstm'))
model.add(Dense(1, name='logit'))
model.add(Activation('sigmoid', name = 'out'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('models/lstm_model.h5')