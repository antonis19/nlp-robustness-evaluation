import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential, Model, Input, load_model
from keras.datasets import imdb
from glove_utils import load_embedding
import pickle
import time
import tensorflow as tf
from data_utils import IMDBDataset
from embedding import Embedding
import utils
from itertools import dropwhile


class IMDBModel:
    '''
    Class representing a model trained on the IMDB sentiment analysis dataset.
    '''
    def __init__(self,model_filename : str, embedding: Embedding, maxlen = 200):
        '''
        Create an IMDBModel class.

        Parameters
        -------------
        model_filename: str
            the path containing the underlying Keras model.
        
        embedding: Embedding
            the embedding space used by the model.

        maxlen: int
            the maximum number of words to pad text to.

        '''
        self.model = load_model(model_filename)
        self.embedding_model = self.create_embedding_model(self.model)
        self.embedding = embedding
        self.word2index = embedding.word2index
        self.index2word = embedding.index2word
        self.index2embedding = embedding.index2embedding
        self.maxlen = maxlen
    

    def sequence_to_embedding(self,seq):
        '''
        Convert sequence of word indexes (rows in embedding matrix) to matrix of embeddings.
        '''
        return np.array([self.index2embedding[index] for index in seq])


    def words_to_sequence(self, words) :
        '''
        Convert list of words to sequence of word indexes.
        '''
        return np.array([self.word2index[word] for word in words])

    def sequence_to_words(self,seq):
        '''
        Convert sequence of word indexes to words.
        '''
        return [self.index2word[idx] for idx in self.unpad_sequence(seq)]

    def seq2text(self,seq):
        '''
        Convert sequence of word indexes to text.
        '''
        return " ".join(self.sequence_to_words(seq))

    def text2seq(self,text, clean_text = True):
        '''
        Convert text to sequence of word indexes.
        '''
        text = IMDBDataset.clean_text(text)
        sample_indexes = IMDBDataset.text_to_index(text, self.word2index)
        sample_indexes = sequence.pad_sequences([sample_indexes], maxlen=self.maxlen, padding = 'pre', truncating = 'pre').squeeze()
        return sample_indexes

    def model_predict(self,x) :
        '''
        Predict probability of positive sentiment from list of word indexes.

        Parameters
        ------------
        x: list
            list of word indexes (rows in embedding matrix) representing the text.

        Returns
        -------------
        out: float
            probability of positive sentiment.
        '''
        out = self.model.predict(np.expand_dims(x, axis=0))[0][0]
        return out

    def model_predict_class(self,x) :
        '''
        Predict the class 1(positive)/ 0(negative) from list of word indexes.
        '''
        out = self.model.predict_classes(np.expand_dims(x, axis=0))[0][0]
        return out

    def predict(self,text):
        '''
        Predict probability of positive sentiment from text.

        Parameters
        ------------
        text: str
            The text whose sentiment we want to predict.

        Returns
        -------------
        out: float
            probability of positive sentiment.
        '''
        indexes = self.text2seq(text)
        return self.model_predict(indexes)
    
    def predict_class(self, text):
        '''
        Predict the class of a given text.
        Parameters
        ------------
        text: str
            The text whose sentiment we want to predict.

        Returns
        -------------
        0 (negative) or 1 (positive)
        '''
        prediction = self.predict(text)
        if prediction < 0.5 :
            return 0
        else :
            return 1

    def preprocess_text(self, text):
        '''
        Preprocess text by cleaning it and padding to maximum length.
        '''
        text = IMDBDataset.clean_text(text)
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text, lower=False, split=' ', filters='\t\n')
        return ' '.join(tokens[-self.maxlen:])

    def unpad_sequence(self, seq, pad_char = 0):
        '''
        Remove pad characters from sequence.
        '''
        return np.array(list(dropwhile(lambda index: index == pad_char, seq)))

    def pad_sequence(self, seq, pad_char = 0):
        '''
        Pad sequence to maximum length.
        '''
        return sequence.pad_sequences([seq], maxlen = self.maxlen, padding = 'pre', truncating= 'pre').squeeze()

    def create_embedding_model(self,model):
        '''
        Get submodel that takes word embeddings as input (instead of word indexes)

        Parameters
        -----------
        model
            The underlying Keras model.

        Returns
        -------------  
        embedding_model
            The Keras model that takes a sequence of word vectors (instead of word indexes)  as input and produces
            the probability of positive sentiment.
        '''
        # Create submodel that takes word embeddings as input, instead of discrete word indexes
        embedding_input = Input(shape = [None, None])
        prediction_layer = embedding_input
        for layer in model.layers[1:]:
            prediction_layer = layer(prediction_layer) # append layer
        embedding_model= Model(inputs= embedding_input, outputs = prediction_layer) # create the submodel
        return embedding_model
    
    def embedding_model_predict(self, word_embeddings):
        '''
        Get the output of the model given a sequence of word embeddings as input.
        
        Parameters
        -------------------
        word_embeddings:
            a sequence of word vectors corresponding to the word embeddings of the sequence of words in the text

        Returns
        ------------------
        Probability of positive sentiment (float)
        '''
        return self.embedding_model.predict(np.expand_dims(word_embeddings, axis=0))[0][0]



if __name__ == '__main__' :
    from glove_utils import load_embedding
    start_time = time.time()
    GLOVE_FILENAME = 'data/glove.6B.100d.txt'
    word2index, index2word, index2embedding = load_embedding(GLOVE_FILENAME)
    print('Loaded %s word vectors in %f seconds' % (len(word2index), time.time() - start_time))
    embedding = Embedding(word2index, index2word, index2embedding)
    imdb_model = IMDBModel('models/lstm_model.h5', embedding)
    text = "Really good movie, highly recommended."
    prediction = imdb_model.predict(text)
    print(prediction)
    seq = imdb_model.text2seq(text)
    word_embeddings = imdb_model.sequence_to_embedding(seq)
    pred = imdb_model.embedding_model_predict(word_embeddings)
    print(pred)
    assert prediction == pred