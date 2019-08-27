import numpy as np
import time
import functools

from IMDBModel import IMDBModel
from embedding import Embedding
from glove_utils import load_embedding
from data_utils import IMDBDataset
from keras.preprocessing import sequence
from utils import *
from optimizer.MultiDimensionalOptimizer import MultiDimensionalOptimizer
from collections import OrderedDict
from pprint import pprint
from copy import deepcopy


class DeepGoTextVerifier :
    '''

    Parameters
    -------------
    model 
        The text classifier under consideration.

    embedding: Embedding
        The embedding space used by the model.

    '''
    def __init__(self, model,embedding: Embedding) :
        self.model = model
        self.embedding = embedding
        # don't consider these tokens in the perturbations
        self.filters = list('!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}') + ['a','an', 'the', 'of']

    def process_text(self,text, N = 4) :
        '''
        Process text to extract sequence of indexes, words, word embeddings and neighbors map.

        Parameters
        ----------------
        text: str
            The input text.

        N : int
            Number of nearest neighbors to consider.

        Returns
        ----------------
        seq: list
            List of word indexes (entries in the embedding matrix) corresponding to the words
            in the text. seq is padded to the model's maximum length.

        words: list
            List of words in the text.
        
        word_embeddings: np.array
            Matrix of word embeddings constituting the intermediate representation of the text.
        
        neighbors_map: dict
            Dictionary mapping each word to a list of its `N` nearest neighbors in the model's embedding space.

        '''
        text = self.model.preprocess_text(text)
        seq = self.model.text2seq(text)
        seq = self.model.pad_sequence(seq)
        words= self.model.sequence_to_words(seq)
        word_embeddings = self.model.sequence_to_embedding(seq)
        neighbors_map = self.embedding.build_neighbors_map(words, N, return_distances = False)
        return seq, words, word_embeddings, neighbors_map
    
    def build_D(self,word, neighbors_words, normalize = True):
        '''
        Build matrix D which contains the perturbation directions towards a word's nearest neighbors in its columns.

        Parameters
        --------------------------
        word : str
            The word that is being perturbed.

        neighbors_words: list
            List of nearest neighbors of `word`.

        normalize: bool
            If set to True the perturbation directions will be normalized to unit vectors.


        Returns
        --------
        D: np.array
            Matrix containing the perturbation directions in its columns.

        '''
        word_embedding = self.embedding.index2embedding[self.embedding.word2index[word]]
        neighbors_embeddings = self.model.sequence_to_embedding(self.model.words_to_sequence(neighbors_words))
        D = neighbors_embeddings - word_embedding
        if normalize : # make perturbation directions unit vectors
            D = normalize_matrix(D)
        return D.T
    
    def build_r(self,D, alphas):
        '''
        Build the perturbation factor r.

        Parameters
        ----------------------
        D: np.array
            Matrix containing the perturbation directions in its columns.

        alphas: np.array
            Array of perturbation weights for each perturbation direction.

        Returns
        ------------------
        r : np.array
            Perturbation vector r =matmul(D,alphas)

        '''
        # print("Before tensordot: ")
        # print("D.shape = ",D.shape)
        # print("alphas.shape = ", alphas.shape)
        r = np.tensordot(D,alphas, axes = (1,0)).squeeze()
        return r

    def substitute(self,input, new_vals, start, length) :
        '''
        Set `input[start:start+length]` to `new_vals` and return it as a new array.
        '''
        assert len(new_vals) == length, 'len(new_vals) = %d, but length = %d' % (len(new_vals), length)
        new_input = deepcopy(input)
        new_input[start:start+length] = new_vals
        return new_input

    def perturb_word(self, index, word_embeddings, word, neighbors, normalize, *alphas):
        '''
        Perturb a word to obtain a perturbed sequence of word embeddings as input to the model.
        A word is perturbed by adding a perturbation factor r to it, which is a linear combination
        of the directions from the word towards its nearest neighbors.

        Parameters
        -----------------------
        index: int
            The position of the word to perturb inside the sequence of word embeddings.

        word: str
            The word to perturb.

        word_embeddings: np.array
            The matrix of word embeddings in the original input.
        
        neighbors: list
            The list of nearest neighbors of `word`.

        normalize: bool
            If set to True, the perturbation directions will be normalized to unit vectors.

        Returns
        --------------
        new_embeddings: np.array
            The perturbed matrix of word embeddings, where row `index` of `word_embeddings` is perturbed,
            while the other rows remain unchanged.


        '''
        alphas = np.array(alphas)
        D = self.build_D(word, neighbors, normalize)
        r = self.build_r(D, alphas).squeeze()
        #print("r.shape = ", r.shape)
        new_embedding = word_embeddings[index] + r
        new_embeddings = self.substitute(word_embeddings.ravel(), new_embedding.ravel(), index*len(new_embedding), len(new_embedding)).\
        reshape(word_embeddings.shape)
        return new_embeddings
    
    def verify_word(self,i, word_embeddings, word, neighbors, normalize = False , epsilon= 1.0, K = None, ETA = None, update_eta = False):
        '''
        Compute the reachability of perturbing a word towards its nearest neighbors.

        Parameters
        ---------------------
        i: int
            Position of word to perturb in the sequence of word embeddings.
        
        word_embeddings: np.array
            The matrix of word embeddings in the original input.

        word: str
            The word to perturb.
        
        neighbors: list
            The list of nearest neighbors of `word`.
        
        normalize: bool
            If set to True, the perturbation directions will be normalized to unit vectors.    

        epsilon: float
            The perturbation magnitude: i.e. the maximum Linfinity norm of the perturbation weights.
        
        K: float, optional
            The Lipschitz constant overestimation to use in the optimization procedure.
        
        ETA: float, optional
            The dynamic Lipschitz constant estimation overshoot factor.

        update_eta : bool
            If set to true, a variable eta will be used, which will be exponentially decayed to `ETA`.



        Returns
        --------------------
        minimum: float
            The minimum confidence in the positive class.

        argmin: list
            The perturbation weights that give the minimum confidence.

        maximum: float
            The maximum confidence in the positive class.
        
        argmax: list
            The perturbation weights that give the maximum confidence.


        '''
        constraints = len(neighbors)* [(0,epsilon)]
        partial_fn = functools.partial(self.perturb_word, i , word_embeddings, word, neighbors, normalize)
        def fn(*alphas) :
            #print("alphas = ", alphas)
            return self.model.embedding_model_predict(partial_fn(*alphas))
        optimizer = MultiDimensionalOptimizer(fn,constraints, ETA  = ETA, K = K, update_eta = update_eta)
        minimum, argmin  = optimizer.minimize()
        maximum, argmax = optimizer.maximize()
        return minimum, argmin, maximum, argmax

        
    def verify_text(self, text, normalize = False, epsilon = 1.0, N = 4,  K = None , ETA = None, update_eta = False):
        '''
        Perform robustness analysis by evaluating the reachability of each word in the text, when the word is perturbed
        towards its nearest neighbors.

        Parameters
        ---------------------
        text: str
            The input text.
        
        normalize: bool
            If set to True, the perturbation directions will be normalized to unit vectors.    

        epsilon: float
            The perturbation magnitude: i.e. the maximum Linfinity norm of the perturbation weights.
        
        K: float, optional
            The Lipschitz constant overestimation to use in the optimization procedure.
        
        ETA: float, optional
            The dynamic Lipschitz constant estimation overshoot factor.

        update_eta : bool
            If set to true, a variable eta will be used, which will be exponentially decayed to `ETA`.



        Returns
        --------------------
        results: OrderedDict
            Ordered dictionary where for each word in the text the reachability of the confidence in the positive class
            has been computed. The results are sorted with least robust words (i.e. words with higher reachability diameter) at the front.
        '''
        seq, words, word_embeddings, neighbors_map = self.process_text(text, N)
        unpadded_seq = self.model.unpad_sequence(seq)
        results = dict() # accumulate results here
        offset = self.model.maxlen - len(unpadded_seq)
        print("Input text: ", text)
        print("seq = ", seq)
        for i, word_index in enumerate(unpadded_seq) :
            print("i = ",i)
            word =  words[i]
            if word_index < 3 : # not a real word
                continue
            if word in self.filters:
                print("Skipping word: ", word)
                continue
            neighbors = [neighbor for neighbor in neighbors_map[word] if neighbors_map[word]!=[]]
            if neighbors == [] :
                print("Skipping word: ",word, " because it has no neighbors.")
                continue
            print("Computing reachability for word: ",[word], " with nearest neighbors: ", neighbors)
            minimum, argmin, maximum, argmax = self.verify_word(offset+i, word_embeddings, word, neighbors, normalize, epsilon, K ,ETA, update_eta)
            results[i] = {
                'word' : word,
                'neighbors': neighbors,
                'min': minimum,
                'max': maximum,
                'rd': maximum-minimum, # reachability diameter
                'argmin': argmin,
                'argmax': argmax,
            }
            print(results[i])
        # sort by descending reachability diameter
        results = OrderedDict(sorted(results.items(), key = lambda x: -x[1]['rd']))
        return results

    def sort_results_by(self,results, key : str, order = 'asc'):
        sign = 1
        if order == 'desc':
            sign = -1
        return OrderedDict(sorted(results.items(), key = lambda x: sign*x[1][key]))


if __name__ == '__main__' :
    # Load GLoVe vectors
    print('Loading GLoVe vectors...')
    start_time = time.time()
    GLOVE_FILENAME = 'data/glove.6B.100d.txt'
    word2index, index2word, index2embedding = load_embedding(GLOVE_FILENAME)
    print('Loaded %s word vectors in %f seconds' % (len(word2index), time.time() - start_time))
    embedding = Embedding(word2index, index2word, index2embedding)

        
    # Load model
    imdb_model = IMDBModel('models/model.h5', embedding)

    # Create Verifier
    verifier = DeepGoTextVerifier(imdb_model, embedding)

    # The text to verify
    text = 'great movie, highly recommended.'

    results = verifier.verify_text(text, normalize= False , epsilon = 1.0, N = 4, K = None , ETA = 2.1, update_eta= True)
    pprint(results)
