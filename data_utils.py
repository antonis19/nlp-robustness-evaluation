"""
Loading and preparing the IMDBDataset. Code mostly based on 
https://github.com/nesl/nlp_adversarial_examples/blob/master/data_utils.py
"""

import os
import re
from collections import Counter
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from glove_utils import load_embedding
import time
import utils


import pickle as pickle
class IMDBDataset(object):
    TRAIN_SET_FILENAME = 'train_set.pickle'
    TEST_SET_FILENAME = 'test_set.pickle'
    def __init__(self, word2index, path='data/aclImdb'):
        self.path = path
        self.train_path = path + '/train'
        self.test_path = path + '/test'
        self.vocab_path = path + '/imdb.vocab'
        self.train_text, self.train_y = self.read_text(self.train_path)
        self.test_text, self.test_y = self.read_text(self.test_path)
        self.train_text = [IMDBDataset.clean_text(text) for text in self.train_text]
        self.test_text = [IMDBDataset.clean_text(text) for text in self.test_text]

        print('tokenizing...')
        
        self.train_seqs = [IMDBDataset.text_to_index(text,word2index) for text in self.train_text]
        self.test_seqs = [IMDBDataset.text_to_index(text, word2index) for text in self.test_text]
        
        print('Dataset built !')

    def clean_text(text):
        return utils.clean_text(text)

    def word_to_index(w, word2index):
        try:
            return word2index[w]
        except KeyError:
            print('Here')
            return 2 # defined to be all zeros

    def text_to_index(text, word2index):
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text, lower=False, split=' ', filters='\t\n')
        return list(map(lambda tok: IMDBDataset.word_to_index(tok, word2index), tokens))


    def save(self, path='data/imdb/'):
        with open(path + IMDBDataset.TRAIN_SET_FILENAME, 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y) , f)

        with open(path + IMDBDataset.TEST_SET_FILENAME, 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y) , f)

    def load_data(path = 'data/imdb/'):       
        with open('data/imdb/train_set.pickle', 'rb') as f:
            train_text, x_train , y_train = pickle.load(f)

        with open('data/imdb/test_set.pickle', 'rb') as f:
            test_text, x_test , y_test = pickle.load(f)
        return ((train_text,x_train,y_train), (test_text, x_test, y_test))

    def read_text(self, path):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        pos_list = []
        neg_list = []
        
        pos_path = path + '/pos'
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_list = [open(x, 'r', encoding = 'utf8').read().lower() for x in pos_files]
        neg_list = [open(x, 'r', encoding = 'utf8').read().lower() for x in neg_files]
        data_list = pos_list + neg_list
        labels_list = [1]*len(pos_list) + [0]*len(neg_list)
        return data_list, labels_list

    # read IMDB's vocabulary
    def read_vocab(vocab_path = 'data/aclImdb/imdb.vocab'):
        with open(vocab_path, 'r', encoding = 'utf8') as f:
            vocab_words = f.read().split('\n')
            return vocab_words

    # get vocabulary after tokenization
    def get_clean_vocab():
        vocab = IMDBDataset.read_vocab()
        # clean vocabulary
        clean_vocab = set()
        for word in vocab :
            clean_text = IMDBDataset.clean_text(word)
            tokens = tf.keras.preprocessing.text.text_to_word_sequence(clean_text, lower=False, split=' ', filters='\t\n')
            [clean_vocab.add(token) for token in tokens]
        clean_vocab = np.array(list(clean_vocab))
        return clean_vocab

    # get vocabulary coverage = known words / vocab_size
    # known words have index != 2
    def get_coverage(vocab, word2index):
        vocab_size = len(vocab)
        indexes = IMDBDataset.text_to_index(" ".join(vocab), word2index)
        indexes = np.array(indexes)
        nonzero_elements = (indexes>2).sum()
        coverage = nonzero_elements / vocab_size
        zero_elements = np.where(indexes == 2)[0]
        unknown_words = vocab[zero_elements]
        return coverage, unknown_words

    

if __name__ == '__main__' :
    GLOVE_FILENAME = 'data/glove.6B.100d.txt'
    start_time = time.time()
    word2index, index2word, index2embedding = load_embedding(GLOVE_FILENAME)
    end_time = time.time() 
    print('Loaded %s word vectors in %f seconds' % (len(word2index), end_time- start_time))
    imdbDataset = IMDBDataset(word2index)
    imdbDataset.save()
    print('Loading data...')
    (train_text, x_train, y_train), (test_text, x_test, y_test) = IMDBDataset.load_data()
