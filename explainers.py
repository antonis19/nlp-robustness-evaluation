import numpy as np
from lime import lime_tabular
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from IMDBModel import IMDBModel
import time
from data_utils import IMDBDataset
from glove_utils import load_embedding
from embedding import Embedding
import keras
from keras.preprocessing import sequence
from utils import get_tokens, preprocess_text
from pprint import pprint
from collections import OrderedDict

# Spectrum-Based Explainer
class SBE :
    '''
    Adaptation of Spectrum-Based Explanations to binary text classification.
    '''

    def __init__(self, model, m = 1000, SIGMA = 2/5):
        '''
        Create an explainer object.

        Parameters
        ------------------
        model
            The text classification model to explain.
        
        m : int 
            Number of mutants to generate.

        SIGMA: float
            Fraction of features to randomly drop in the text.

        '''
        self.model = model
        # hyperparameters
        self.SIGMA = SIGMA # fraction of features dropped
        self.m = m  # number of mutants to be generated from the input instance

    def Tarantula_M(a_ep, a_ef, a_np, a_nf) :
        '''
        Tarantula suspiciousness measure function.
        '''
        suspiciousness = 0
        if a_ef == 0 :
            numerator = 0
        else: 
            numerator  =  a_ef / (a_ef + a_nf)
        if a_ep == 0 :
            denominator = numerator
        else :
            denominator =  numerator + (a_ep / (a_ep + a_np))
        if denominator == 0 :
            suspiciousness = 0
        suspiciousness = numerator / denominator
        return suspiciousness

    def print_stats(a_ep, a_ef, a_np, a_nf):
        for i in range(len(a_ep)):
            print(f"i = {i} ,  a_ep = {a_ep[i]}, a_ef = {a_ef[i]}, a_np = {a_np[i]}, a_nf= {a_nf[i]}")


    def sbe(self, x, y, rank_by_importance = True) :
        '''
        Main algorithm that ranks the tokens in `x`.

        Parameters
        -------------------
        x: list
            List of tokens forming the input text. 
            The tokens could be words, phrases or sentences.
        
        y: int
            The class label of `x`.

        rank_by_importance: bool
            If set to true, the tokens will be ranked based on their importance/polarity.
            If set to false, tokens will be ranked from most negative to most positive.
        
        Returns
        ---------------
        token_ranking: list
            Ranking of token indexes (i.e. indexes of the `x` list).

        values: list
            List of values for each token in `x`.
            The could be importance values (if `rank_by_importance` is set to True),
            or weights (if `rank_by_importance` is set to False).
        '''
        m = self.m
        M  = SBE.Tarantula_M # suspiciousness measure function
        n = len(x)
        sigma = self.SIGMA
        proportion = int(n*sigma) # number of words to drop
        a_eps, a_efs, a_nps, a_nfs =  (np.array(n*[0.0]), np.array(n*[0.0]), np.array(n*[0.0]), np.array(n*[0.0]))
        for counter in range(m) :
            masked_indexes = np.random.choice(n,proportion, replace = 'false')
            unmasked_indexes = np.setdiff1d(np.arange(0,n), masked_indexes)
            unmasked_tokens = np.array(x)[unmasked_indexes].tolist()
            mutant_text = ' '.join(unmasked_tokens)
            prediction = self.model.predict(mutant_text)
            a_eps[unmasked_indexes]+= prediction
            a_nps[masked_indexes]+= prediction
            a_efs[unmasked_indexes]+= 1 - prediction
            a_nfs[masked_indexes]+= 1 - prediction
        assert abs(m - (a_eps[0]+a_efs[0] + a_nps[0] + a_nfs[0])) < 0.0001  , "sum = %f m = %f" % (a_eps[0] + a_efs[0] + a_nps[0] + a_nfs[0], m)
        values = np.array([M(a_eps[i], a_efs[i], a_nps[i], a_nfs[i]) for i in range(n)])
        values =  0.5 - values
        if rank_by_importance:
            values = np.abs(values)
        if rank_by_importance:
            token_ranking = np.argsort(-values)
        else :
            token_ranking = np.argsort(values)
        return token_ranking, values

    def explain_text_words(self,text, rank_by_importance = True):
        '''
        Word level explanation.
        '''
        text = preprocess_text(text)
        text_words = get_tokens(text)
        y = self.model.predict_class(text)
        word_ranking, values = self.sbe(text_words ,y, rank_by_importance)
        ranked_words =  [text_words[i] for i in word_ranking]
        return word_ranking, ranked_words, values

    def explain_tokens(self, tokens, rank_by_importance = True):
        '''
        Wrapper around `sbe` method. 
        
        Parameters
        ---------------
        tokens: list
            List of tokens (words, phrases, or sentences) forming the text. 

        rank_by_importance: bool
            Set to True if tokens should be ranked by polarity, 
            set to False to rank tokens from most negative to most positive.

        Returns
        -----------------
        tokens_ranking: list
            Ranking of token indexes (i.e. indexes of the `x` list).

        ranked_tokens: list
            List of tokens, as ranked by the `sbe` method.

        values: list
            List of values for each token in `x`.
            The could be importance values (if `rank_by_importance` is set to True),
            or weights (if `rank_by_importance` is set to False).

        '''
        text = ' '.join(tokens)
        y = self.model.predict_class(text)
        tokens_ranking, values = self.sbe(tokens,y, rank_by_importance)
        ranked_tokens = [tokens[i] for i in tokens_ranking]
        return tokens_ranking, ranked_tokens, values

    def explain(self, text, explanation_size):
        '''
        Wrapper around `explain_text_words_method`.

        Parameters
        --------------
        text: str
            The text to explain.
        
        explanation_size: int
            The number of top-ranked words to return as explanation.
        
        Returns
        -------------
        word_ranking : list
            Indexes of the `explanation_size` top-ranked words in the text.
        
        ranked_words: list
            List of `explanation_size` top-ranked words in the text.
        '''
        word_ranking, ranked_words, _ = self.explain_text_words(text)
        return word_ranking[:explanation_size], ranked_words[:explanation_size]


class LIMEExplainer :
    '''
    Wrapper around `LIMETextExplainer`, for binary text classification.
    '''
    def __init__(self, model, nsamples = 1000):
        '''
        
        Parameters
        -------------
        model
            The text classification model to explain.
        
        nsamples: int
            The number of neighborhood samples to generate around an input text.

        '''
        self.model = model
        self.nsamples = nsamples

    
    def predict_texts(self, sample_texts):
        '''
        Function that predicts the probability of negative class, and probability of positive class, for
        a list of texts using the model to explain.

        Parameters
        -------------
        sample_texts: list
            List of texts to predict. These are the neighborhood samples generated by LIME.


        Returns
        -----------
        List of tuples (negative_prob, positive_prob) of predicted probabilities.

        '''
        predictions = np.array([np.array([1-self.model.predict(text),self.model.predict(text)]) for text in sample_texts])
        return predictions.reshape(len(predictions),2)
    
    def explain(self, text, nwords, return_weights = False) :
        '''
        Use `LimeTextExplainer` to obtain the top `nwords` most important/polar words in the `text` as 
        an explanation.


        Parameters
        --------------
        text: str
            The text to explain.

        nwords: int
            The number of most important words to return (i.e. explanation size).

        return_weights: bool
            Set to True to return the weights assigned by LIME also.

        Returns
        ---------------
        word_ranking : list
            Indexes of the `nwords` top-ranked words in the text.
        
        ranked_words: list
            List of `nwords` top-ranked words in the text.

        weights: dict, optional
            The dictionary of weights (wordposition -> weight) assigned by LIME to the words
            in the text.

        explanation: optional
            The explanation object returned by `LimeTextExplainer`.
        '''
        text = preprocess_text(text)
        text_words = get_tokens(text)

        class_names = ['negative', 'positive']
        # bow is set to False because word order is important
        explainer = LimeTextExplainer(class_names= class_names, feature_selection = 'auto', bow = False,
         split_expression = ' ', verbose = False)

        explanation = explainer.explain_instance(text_instance = text, labels= [0,1] ,
                         classifier_fn= self.predict_texts, num_features= nwords, num_samples= self.nsamples)
        # sort weights by decreasing absolute value
        weights = OrderedDict(sorted(explanation.as_map()[1], key = lambda weight : - abs(weight[1]) ) )
        word_ranking = np.array(list(weights.keys()))
        ranked_words = [text_words[i] for i in word_ranking]
        if return_weights:
            return word_ranking, ranked_words, weights, explanation
        return word_ranking, ranked_words
    

    def plot_weights(self,lime_weights, explanation):
        '''
        Plot the weights generated by `LimeTextExplainer`.

        Parameters
        -------------
        lime_weights: dict
          The dictionary of weights (wordposition -> weight) assigned by LIME to the words
            in the text.

        explanation
            The explanation object returned by `LimeTextExplainer`.
        '''
        mp = explanation.domain_mapper.map_exp_ids(lime_weights.items())
        words = [word for (word, weight) in mp]
        weights = np.array([weight for (word,weight) in mp])  
        sorted_indexes = np.argsort(-weights)
        words = [words[i] for i in sorted_indexes]
        weights = weights[sorted_indexes]
        df = pd.DataFrame({'words': words, 'weights': weights})
        # plot
        colors = ['r' if weight < 0 else 'g' for weight in df.weights]
        sns.set_color_codes("pastel")
        ax = sns.barplot(y=df.index, x ="weights", data=df, palette=colors, orient = 'h', ci = None)
        ax.set_yticklabels(df['words'])
        ax.set_ylabel('words')
        return ax

if __name__ == '__main__' :
    # Load GLoVe vectors
    print('Loading GLoVe vectors...')
    start_time = time.time()
    GLOVE_FILENAME = 'data/glove.6B.100d.txt'
    word2index, index2word, index2embedding = load_embedding(GLOVE_FILENAME)
    print('Loaded %s word vectors in %f seconds' % (len(word2index), time.time() - start_time))
    embedding = Embedding(word2index, index2word, index2embedding)

    # Load model
    imdb_model = IMDBModel('models/lstm_model.h5', embedding)

    # Load data
    maxlen = 200
    batch_size = 32
    print('Loading data...')
    (train_text, x_train, y_train), (test_text, x_test, y_test) = IMDBDataset.load_data()
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding = 'pre', truncating = 'pre')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding = 'pre', truncating = 'pre')
    print('Data loaded.')

    text = "This movie is absolutely incredible."
    print("Text: ")
    print(text)

    print("Running SBE...")
    sbe = SBE(imdb_model, m = 1000)
    words_ranking , ranked_words = sbe.explain(text,4)
    print("Word ranking: ", words_ranking)
    print("Most important words: ", ranked_words)

    print("Running LIME...")
    lime_explainer = LIMEExplainer(imdb_model, nsamples = 1000)
    word_ranking, ranked_words, weights, explanation = lime_explainer.explain(text, 5, return_weights= True)
    print("word_ranking = ", word_ranking)
    print("ranked_words = ", ranked_words)
    lime_explainer.plot_weights(weights,explanation)

