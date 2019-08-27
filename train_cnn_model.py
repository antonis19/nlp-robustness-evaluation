'''
Trains an CNN model on the IMDB sentiment classification task.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM
from keras.datasets import imdb
from data_utils import IMDBDataset
from glove_utils import load_embedding
import time

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
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



print('Build model...')
model = Sequential()
# the +3 is because positions 0 1 and 2 are for special characters like <PAD>, <START> and <UNK>
model.add(Embedding(len(index2embedding), EMBEDDING_DIM, weights=[index2embedding], trainable = False, name = 'embedding'))
model.add(Conv1D(filters=100, kernel_size=5, padding='valid', activation='tanh', strides=1, name = 'cnn'))
model.add(GlobalMaxPooling1D(name = 'pooling'))
model.add(Dense(64, activation = 'relu', name = 'dense'))
model.add(Dropout(0.5, name = 'dropout'))
model.add(Dense(1, name ='logit')) 
model.add(Activation('sigmoid', name = 'out'))

model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train for 2 epochs to get 73% accuract
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('models/cnn_model.h5')