import numpy as np
import tensorflow as tf
from IMDBModel import IMDBModel
from embedding import Embedding
import time
from glove_utils import load_embedding, load_syn_dict, load_dist_dict
from data_utils import IMDBDataset
from pprint import pprint
from collections import OrderedDict
from copy import deepcopy
from utils import  preprocess_text, get_tokens

class Attacker :
    '''
    Class that attacks model to generate adversarial examples, or change the classification to the correct class.

    Parameters
    ----------
    model
       The model to attack. Should implement methods predict and predict_class.

    synonyms_embedding : Embedding
       The embedding space to generate nearest neighbors for the attack.

    explainer
       Explainer that targets the most important words in a text for replacement. Implements the explain method.
       Use None if you do not want to use an explainer.

    tagger
       A part-of-speech tagger. Implements the get_tag_list method to get POS tags from text.

    percentage: float
        The percentage of words to target for replacement in a text.
    
    neighborhood_size: int
        The number of nearest neighbors to consider for the attack.
    
    max_distance: 
        The maximum allowed distance between a word and its neighbor. Neighbors with distance greater than `max_distance`
        from a word will be discarded as not semantically similar enough.

    syn_dict_path:  str, optional
        The path to cached nearest neighbors map  {word: neighbors}.
    
    dist_dict_path: str, optional
        The path to cached distances of nearest neighbors {word: neighbor_distances}.

    '''
    def __init__(self, model, synonyms_embedding, explainer , tagger, percentage = 0.3, neighborhood_size = 10,  max_distance = None,
    syn_dict_path = None, dist_dict_path = None):
        self.model = model
        self.synonyms_embedding = synonyms_embedding
        self.explainer = explainer
        self.tagger = tagger
        self.percentage = percentage
        self.neighborhood_size = neighborhood_size
        self.max_distance = max_distance
        if syn_dict_path is None: 
            self.syn_dict = dict()
        else:
            self.syn_dict = load_syn_dict(syn_dict_path, N = self.neighborhood_size)
        if dist_dict_path is None:
            self.dist_dict = dict()
        else :
            self.dist_dict = load_dist_dict(dist_dict_path, N = self.neighborhood_size)

    def filter_synonyms(self, x, index, synonyms):
        '''
        Syntactically filter the `synonyms` of word at position `index` of `x`. 
        Words in `synonyms` that are not the same part of speech as `x[index]` are filtered out.
        
        Parameters
        ----------
        x: list
           The list of (tokenized) words forming the text.

        index: int
            The position of the word in `x`.

        synonyms: list
            The list of synonyms for word `x[index]` to syntactically filter.


        Returns
        ---------
        valid_synonyms:
            List consisting of words in `synonyms` that are the same part of speech as `x[index]` 
            in the context of the words `x` in the text.


        '''
        x_text = ' '.join(x)
        original_tags = self.tagger.get_tag_list(x_text)
        valid_synonyms = []
        for synonym in synonyms :
            modified_x = deepcopy(x)
            modified_x[index] = synonym
            modified_x_text = ' '.join(modified_x)
            modified_tags = self.tagger.get_tag_list(modified_x_text)
            if original_tags == modified_tags:
                valid_synonyms.append(synonym)
        return valid_synonyms
    
    def get_valid_replacements(self, x, indexes, synonyms_map):
        '''
        Syntactically filter each of the words at positions `indexes` of  word list `x` forming the text.


        Parameters
        ------------
        x : list
            The list of (tokenized) words forming the text.

        indexes: list
            List of word positions in x. 

        synonyms_map: dict
            A dictionary mapping each word in the text to a list of its nearest neighbors.


        Returns
        ------------
        candidate_replacements: dict
            A dictionary of the form {position: {word: word, replacements: synonyms}} consisting the
            valid replacements for each word position from `indexes`.

        '''
        candidate_replacements = dict()
        for index in indexes :
            word = x[index]
            synonyms = synonyms_map[word]
            synonyms  = self.filter_synonyms(x, index, synonyms)
            if synonyms == [] :
                continue
            candidate_replacements[index] = {
                "word" : word,
                "replacements" : synonyms
            }
        return candidate_replacements

    '''
    Print the stats of candidate replacements

    Parameters
    -----------
    candidate_replacements: dict
        {pos: (word, replacement_words)}
        A dictionary mapping each word position in the text to a list of semantically similar and
        syntactically equivalent words.

    '''
    def print_candidate_stats(candidate_replacements):
        nwords = len(candidate_replacements.keys())
        if nwords == 0 :
            print("No cadidate replacements.")
            return
        nreplacements = 0
        for (pos, entry) in candidate_replacements.items():
            nreplacements +=  len(entry['replacements'])
        print("number of candidate words: ", nwords)
        print("number of possible replacements: ", nreplacements)
        print("average number of replacements per word: ", nreplacements/nwords)


    def predict_replacements(self, x, candidate_replacements):        
        '''
        Get the predicted probabilities of applying candidate_replacements.

        Parameters
        -------------

        x: list
            list of words forming the text.

        candidate_replacements: dict
            candidate replacements for the attack.

        Returns
        -------------
        predicted_replacements: OrderedDict
            `candidate_replacements` augmented with an entry 
            with the resulting probabilities of applying each replacement in it.


        '''
        predicted_replacements = deepcopy(candidate_replacements)
        for (pos, entry) in candidate_replacements.items():
            replacements = entry['replacements']
            predictions = []
            for replacement in replacements:
                adv_x = deepcopy(x)
                adv_x[pos] = replacement
                adv_x_text = ' '.join(adv_x)
                prediction = self.model.predict(adv_x_text)
                predictions.append(prediction)
            predicted_replacements[pos]['probs'] = predictions
        return OrderedDict(predicted_replacements)

    
    def get_best_replacement(self, predicted_replacements, target_class):
        '''
        Get the replacement that increases the confidence in `target_class` the most.

        Parameters
        --------------------
        
        predicted_replacements: dict
            candidate replacements map, augmented with prediction probabilities of applying them.

        target_class: int
            The label of the target class .


        Returns
        --------------------
        (best_pos,best_word,best_replacement)  .


        '''
        best_pos = -1; best_word = ''; best_replacement = ''
        best_difference = np.inf
        for (pos, entry) in predicted_replacements.items():
            word = entry['word']
            replacements = entry['replacements']
            probs = entry['probs']
            for (i,replacement) in enumerate(replacements):
                prob = probs[i]
                if abs(target_class - prob) < best_difference:
                    best_difference = abs(target_class - prob)
                    best_pos = pos
                    best_word = word
                    best_replacement = replacements[i]

        return (best_pos,best_word,best_replacement)


    
    def get_replacements_as_list(self, predicted_replacements):
        '''
        Return the predicted replacements as a list of elements (pos,word,replacement, prob) .
        '''
        replacements_list = []
        for (pos, entry) in predicted_replacements.items():
            word = entry['word']
            replacements = entry['replacements']
            probs = entry['probs']
            assert len(replacements) == len(probs)
            for i in range(len(replacements)):
                replacements_list.append((pos, word, replacements[i], probs[i]))
        return replacements_list
    
    def rank_replacements(self, predicted_replacements, target_class):
        '''
        Rank `predicted_replacements` in order of decreasing distance to `target_class`.
        '''
        replacements_list = self.get_replacements_as_list(predicted_replacements)
        return sorted(replacements_list, key = lambda tup : abs(target_class - tup[3]))
    
    def get_best_n_replacements(self, predicted_replacements, n, target_class):
        '''
        Get the `n` predicted replacements  with the least distance to `target_class`.
        '''
        ranked_replacements = self.rank_replacements(predicted_replacements, target_class)
        return ranked_replacements[:n]

    def greedy_search(self, x, candidate_replacements, target_class):
        '''
        Apply greedy search on the `candidate_replacements` in order to pick the ones
        that change the class text to `target_class`.

        Parameters
        --------------------

        x: list
            List of words in forming the text.
        
        candidate_replacements: dict
            A dictionary of the form {position: {word: word, replacements: synonyms}} consisting the
            valid replacements for each word position considered for replacement.

        target_class: int
            The label of the target class


        Returns
        -------------------
        used_replacements: list
        A list of tuples (position, word, replacement_word) 
        '''
        candidate_replacements = deepcopy(candidate_replacements)
        used_replacements = []
        adversary_found = False
        prediction = self.model.predict(' '.join(x))
        while not adversary_found and candidate_replacements != {} :
            predicted_replacements = self.predict_replacements(x, candidate_replacements)
            (pos, word, replacement) = self.get_best_replacement(predicted_replacements, target_class)
            used_replacements.append((pos,word,replacement))
            adv_x = deepcopy(x)
            adv_x[pos] = replacement
            adv_x_text = ' '.join(adv_x)
            predicted_class = self.model.predict_class(adv_x_text)
            prediction = self.model.predict(adv_x_text)
            x = adv_x  # apply replacement
            del candidate_replacements[pos]
            if predicted_class == target_class:
                adversary_found = True
                break
        # sort replacements by word position
        used_replacements = sorted(used_replacements, key = lambda x : x[0])
        return used_replacements, adversary_found, prediction

    def beam_search(self, x, candidate_replacements , target_class, beam_size = 4, return_multiple = False):
        '''
        Apply beam search on `candidate_replacements` to change the classification of the text to `target_class`.

        Parameters
        ----------------
        x: list
            List of words forming the text.

        candidate_replacements: dict
            The search space consisting of the candidate replacements to search in order to
            change the classification.

        target_class: int
            The label of the target class.

        beam_size: int

        return_multiple: bool
            Boolean flag that returns a set of suggested replacement lists that
            change the classification  to `target_class` .

        Returns
        ----------------
        used_replacements or list of used replacements if `return_multiple`
        '''
        # nodes of the search tree are tuples (x, candidate_replacements, used_replacements )
        this_level_nodes = [(x,candidate_replacements, [])]
        level = 0
        while this_level_nodes != []:
            assert(len(this_level_nodes) <= beam_size), "ERROR: %d nodes in this level" % (len(this_level_nodes))
            next_level_nodes = []
            visited_replacements = set()
            # contains tuples (node_nr, pos, word, replacement, prob) out of which,
            # we select the best beam_size tuples
            level_best_replacements = [] 
            for i in range(len(this_level_nodes)) :
                _x, _candidate_replacements, _used_replacements = deepcopy(this_level_nodes[i])
                _predicted_replacements = self.predict_replacements(_x, _candidate_replacements)
                _ranked_replacements = self.rank_replacements(_predicted_replacements, target_class)
                _ranked_replacements = list(filter(lambda replacement: 
                 frozenset(_used_replacements + [(replacement[0],replacement[1],replacement[2])]) not in visited_replacements, _ranked_replacements))
                _best_replacements = _ranked_replacements[:beam_size]
                _best_replacements_indexed = [(i,*replacement) for replacement in _best_replacements]
                level_best_replacements.extend(_best_replacements_indexed)
                
                possible_replacements = [frozenset(_used_replacements + [(replacement[0],replacement[1],replacement[2])]) for replacement in _best_replacements]
                visited_replacements.update(possible_replacements)
            for i in range(len(this_level_nodes)):
                _x, _candidate_replacements, _used_replacements = deepcopy(this_level_nodes[i])
            level_best_replacements = level_best_replacements[:beam_size]
            suggestions= []
            # next step: apply the replacements
            for _best_replacement in level_best_replacements:
                node_index,  pos,  word, replacement_word, prob = _best_replacement
                _x, _candidate_replacements, _used_replacements = deepcopy(this_level_nodes[node_index])
                adv_x = deepcopy(_x)
                adv_x[pos] = replacement_word
                adv_x_text = ' '.join(adv_x)
                predicted_class = self.model.predict_class(adv_x_text)
                prediction = self.model.predict(adv_x_text)
                _used_replacements.append((pos,word, replacement_word))
                del _candidate_replacements[pos]
                assert prediction == prob
                if predicted_class == target_class: # adversarial example found
                    if not return_multiple :
                        return _used_replacements
                    else :
                        suggestions.append((_used_replacements, prediction))
                next_level_nodes.append((adv_x, _candidate_replacements, _used_replacements))
            if return_multiple and len(suggestions) > 0 :
                return suggestions
            this_level_nodes = next_level_nodes
            level+=1
        if  return_multiple :
            return []
        # Failed to find adversarial example, return results from a node
        _x, _candidate_replacements, _used_replacements
        _x_text = ' '.join(_x)
        return _used_replacements



    def get_adv_text(orig_text, used_replacements):
        '''
        Apply replacements to text to obtain adversarial text.
        '''
        text_words = get_tokens(orig_text)
        for (pos, word, replacement_word) in used_replacements:
            assert text_words[pos] == word, 'pos = %d, text_word = %s , word = %s' % (pos, text_words[pos], word)
            text_words[pos] = replacement_word
        return ' '.join(text_words)


    def build_synonyms_map(self, candidate_words):
        '''
        Build a map {word: synonyms} for each word in `candidate_words`, where synonyms are nearest neighbors of each word
        within distance `self.max_distance`       
        '''
        uncached_words = [word  for word in candidate_words if word not in self.syn_dict]
        cached_words = [word for word in candidate_words if word in self.syn_dict]
        cached_synonyms_map = {word: self.syn_dict[word] for word in cached_words}
        cached_dist_map = {word: self.dist_dict[word] for word in cached_words}
        uncached_synonyms_map, uncached_dist_map = self.synonyms_embedding.build_neighbors_map(uncached_words, N = self.neighborhood_size,
        return_distances = True)
        synonyms_map = {**cached_synonyms_map, **uncached_synonyms_map}
        dist_map = {**cached_dist_map, **uncached_dist_map}
        if self.max_distance is not None:
            synonyms_map = Embedding.filter_by_distance(synonyms_map, dist_map, self.max_distance)
        return synonyms_map

    def attack(self,text, target_class, search_algorithm, random_attack = False):
        '''
        Attack text to change the prediction to `target_class`.

        Parameters
        -----------------
        text: str
            The text to attack.
        
        target_class: int
            The class to change the classification to.

        search_algorithm: str
            The search algorithm to use in attack the text : greedy or beam.

        random_attack: bool, optional
            Randomly selects words to target for attack

        '''
        text = preprocess_text(text)
        x = get_tokens(text)
        explanation_size = int(self.percentage * len(x))
        if self.explainer is None : # target all words
            print("No explainer provided .  Targeting all words in the input ... ")
            candidate_words_indexes = np.arange(len(x))
            candidate_words = np.array(x)[candidate_words_indexes].tolist()
        elif not random_attack :
            print('Generating explanation...')
            candidate_words_indexes, candidate_words = self.explainer.explain(text, explanation_size)
        else :
            print("Randomly selecting candidate words to perturb...")
            candidate_words_indexes = np.random.choice(len(x), explanation_size , replace = False)
            candidate_words = np.array(x)[candidate_words_indexes].tolist()
        assert len(candidate_words_indexes) == len(candidate_words)
        print("Extracted candidate words: ", candidate_words)
        synonyms_map = self.build_synonyms_map(candidate_words)
        print("Built synonyms map.")
        candidate_replacements = self.get_valid_replacements(x, candidate_words_indexes, synonyms_map)
        print("Filtered replacements.")
        Attacker.print_candidate_stats(candidate_replacements)
        #print("candidate_replacements: ")
        #pprint(candidate_replacements)
        if search_algorithm == 'greedy':
            print('Running greedy search...')
            used_replacements, adversary_found, prediction = self.greedy_search(x,candidate_replacements, target_class)
        elif search_algorithm == 'beam':
            print('Running beam search...')
            used_replacements, adversary_found, prediction = self.beam_search(x, candidate_replacements, target_class)
        else :
            raise ValueError('Invalid search algorithm provided')
        print("Chose replacements.")

        # Generate adversarial text
        adv_text = Attacker.get_adv_text(text, used_replacements)
        return used_replacements, adversary_found, adv_text, prediction


    def fix(self, text, target_class, beam_size = 4, random_fix = False):
        '''
        Change the classification of a text to the correct class.

        Parameters
        ------------
        text: str
            The text that is misclassified.
        
        target_class: int
            The label of the class to change the prediction to

        beam_size: int

        random_fix: Boolean, Optional
            If set to True, words will be targeted randomly for replacement.


        Returns
        ----------------
        suggestions: list
            The list of suggested replacement sets.


        '''
        text = preprocess_text(text)
        x = get_tokens(text)
        explanation_size = int(self.percentage * len(x))
        if self.explainer is None : # target all words
            print("No explainer provided .  Targeting all words in the input ... ")
            candidate_words_indexes = np.arange(len(x))
            candidate_words = np.array(x)[candidate_words_indexes].tolist()
        elif not random_fix :
            print('Generating explanation...')
            candidate_words_indexes, candidate_words = self.explainer.explain(text, explanation_size)
        else :
            print("Randomly selecting candidate words to perturb...")
            candidate_words_indexes = np.random.choice(len(x), explanation_size , replace = False)
            candidate_words = np.array(x)[candidate_words_indexes].tolist()
        print("Extracted candidate words: ", candidate_words)
        synonyms_map = self.build_synonyms_map(candidate_words)
        print("Built synonyms map.")
        candidate_replacements = self.get_valid_replacements(x, candidate_words_indexes, synonyms_map)
        print('Filtered replacements.')
        print('Running beam search...')
        suggestions = self.beam_search(x, candidate_replacements, target_class, beam_size = beam_size, return_multiple = True)
        return suggestions


                
            

