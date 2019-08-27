from textblob import TextBlob

'''
Part-Of-Speech Taggers.

TextBlobTagger is more efficient but less accurate than SpacyTagger.
'''

class TextBlobTagger :
    '''
     A wrapper around the NLTK tagger textblob uses.
    '''
    def __init__(self):
        pass

    def get_tag_list(self, text) :
        '''
        Get list of tags from text.
        '''
        blob = TextBlob(text)
        tags = [tag for (word,tag) in blob.tags]
        return tags

    def get_ngram_tags(self, text, n = 3) :
        ngrams = TextBlob(text).ngrams(n)
        return [self.get_tag_list(" ".join(ngram)) for ngram in ngrams]

    
    def translate_pos(self, original_tokens, new_tokens, pos):
        new_pos = 0
        for (i, token) in enumerate(original_tokens) :
            if new_pos >= len(new_tokens):
                return -1
            if i > pos :
                return -1
            elif pos == i :
                if  token != new_tokens[new_pos]:
                    return -1
                return new_pos
            else :
                if token == new_tokens[new_pos]:
                    new_pos+=1
                else :
                    pass

    # convert word to the TextBlob interpretation of it
    def to_text_blob_word(self,word):
        singleton_words = TextBlob(word).words
        if len(singleton_words) == 0 : # TextBlob does not consider it a word
            return word
        else :
            return singleton_words[0]

    def get_ngram_window(self,tokens, pos, n = 3):
        # map words to TextBlob format
        word_tokens = [self.to_text_blob_word(word_token) for word_token in tokens]
        indexed_word_tokens = [(i,token) for (i, token) in enumerate(word_tokens)]
        text = TextBlob(" ".join(word_tokens))
        words = text.words
        indexed_words = [(i, token) for (i, token) in enumerate(words)]
        new_pos = self.translate_pos(word_tokens, words, pos)
        if new_pos == -1 :
            return []
        window_start = max(0, new_pos - n + 1)
        window_end = min(new_pos + n, len(words))
        return words[window_start:window_end]

    def get_ngram_tags_at_pos(self, tokens, pos, n = 3):
        ngram_window = self.get_ngram_window(tokens, pos, n)
        text = " ".join(ngram_window)
        return self.get_ngram_tags(text, n)




import spacy
class SpacyTagger():
    '''
    spaCy's tagger.
    '''
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def get_tag_list(self, text):
        '''
        Get list of tags from text.
        '''
        tags = [tag for (word,tag) in self.tags(text)]
        return tags
   
    def tags(self, text):
        tokens = self.nlp(text)
        tags = []
        for token in tokens:
            tags.append((token, token.tag_))
        return tags




if __name__ == '__main__' :
    orig = "it drew you into the players and emotion of the game ."
    adv = "it drew you into the players and sympathetic of the game . "
    adv2 = "it drew you into the players and passion of the game . "
    pos =  7
    spacy_tagger = SpacyTagger()
    adv_tags = spacy_tagger.tags(adv)
    # print("orig_tags = ", orig_tags)
    print("adv_tags = ", adv_tags)
    # print("adv2_tags = ", adv2_tags)