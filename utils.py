from numpy import linalg as LA
import numpy as np

def l2_norm(x):
    '''
    The L2 of vector `x`.
    '''
    return LA.norm(x,2)

def linf_norm(x):
    '''
    L infinity norm of vector `x`.
    '''
    return LA.norm(x,np.inf)


def normalize_matrix(M):
    '''
    normalize matrix row-wise, so that rows are unit vectors.
    '''
    return np.apply_along_axis(normalize_vector, 1, M)

    
def normalize_vector(x):
    '''
    normalize vector to unit vector.
    '''
    norm = l2_norm(x)
    if norm == 0:
        return x
    return x / norm


# map a vector in L-inf ball  to L2 norm ball
def linf_to_l2(v) :
    if not np.any(v) : # all zeros
        return v
    return (linf_norm(v) /l2_norm(v)) * v

def l2_to_linf(v):
    if not np.any(v): # zero vector
        return v
    return (l2_norm(v)/ linf_norm(v)) * v


# Preprocessing utils
import re
import keras

# https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Spacing out punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<[^A-Z]*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()

    # space out punctuation
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in filters}))
    return text

def get_tokens(text, split_expr = ' '):
    '''
    Get word tokens from text.
    '''
    text = clean_text(text)
    tokens = keras.preprocessing.text.text_to_word_sequence(text, lower=True, split=split_expr, filters='\t\n')
    return tokens

def preprocess_text(text):
    tokens = get_tokens(text)
    return ' '.join(tokens)
