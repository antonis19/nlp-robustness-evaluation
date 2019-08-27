from stanfordcorenlp import StanfordCoreNLP
from nltk import Tree, ParentedTree

class StanfordParser :
    STANFORD_CORE_NLP_PATH = 'stanford-corenlp-full-2018-10-05'
    def __init__(self, parser_path = STANFORD_CORE_NLP_PATH):
        self.parser_path = parser_path
        self.nlp = StanfordCoreNLP(self.parser_path)


    def get_parse_string(self, sentence):
        result_string = ''
        parse_string = self.nlp.parse(sentence)
        print("parse string: ", parse_string)
        lines = parse_string.splitlines()
        for line in lines :
            result_string+=line
        return result_string

    def build_phrases(self, tree, current_phrases, split_tags = ['S', 'SBAR', '.']):
        if type(tree) == str : # leaf node
            if tree != '.' :
                current_phrases[-1]+=str(tree)+' '
            return
        if tree.label() in split_tags: # start new phrase here
            if tree.label() == '.' :
                current_phrases[-1]+='.'
            current_phrases.append('')
        for child in tree:
            self.build_phrases(child, current_phrases, split_tags)


    def get_phrases(self, sentence, split_tags = ['S', 'SBAR', '.']):
        '''
        Split sentence into a list of phrases.

        Parameters
        --------------
        
        sentence: str
            The sentence to split into phrases.

        split_tags: list
            The list of labels in the parse tree to split the sentence at.
        '''
        phrases = []
        parse_str = self.get_parse_string(sentence)
        t = Tree.fromstring(parse_str)
        assert t.label() == 'ROOT'
        children = []
        for child in t:
            children.append(child)
        assert len(children) == 1 
        t = children[0]
        if t.label() not in split_tags :
            return [' '.join(t.leaves())]
        phrases = []
        self.build_phrases(t, phrases, split_tags)
        # remove empty strings and trailing spaces
        phrases = [phrase for phrase in phrases if phrase != '']
        #print("Extracted phrases: ", phrases)
        # t.draw()
        return phrases


    def close(self):
        '''
        Shutdown the backend server running the parser.
        '''
        self.nlp.close()


if __name__ == '__main__':
    parser = StanfordParser()
    sentence = 'this is a cross between the last don and godfather 2 . '
    phrases = parser.get_phrases(sentence)
    print(phrases)