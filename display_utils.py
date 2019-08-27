from IPython.core.display import display, HTML
import numpy as np
from utils import get_tokens

#  Based on https://github.com/nesl/nlp_adversarial_examples/blob/master/display_utils.py
def html_render(orig_text, adv_text):
    orig_text_words = orig_text.split(' ')
    adv_text_words = adv_text.split(' ')
    orig_html = []
    adv_html = []
    assert(len(orig_text_words) == len(adv_text_words)), "%d words in original, but %d words in adversarial text" % \
    (len(orig_text_words), len(adv_text_words))
    for i in range(len(orig_text_words)):
        if orig_text_words[i] == adv_text_words[i]:
            orig_html.append(orig_text_words[i])
            adv_html.append(adv_text_words[i])
        else:
            orig_html.append(format("<b style='color:red'>%s</b>" %orig_text_words[i]))
            adv_html.append(format("<b style='color:blue'>%s</b>" %adv_text_words[i]))
    
    orig_html = ' '.join(orig_html)
    adv_html = ' '.join(adv_html)
    return orig_html, adv_html

def display_html(html_text):
    '''
    Display HTML in Jupyter notebook.
    '''
    display(HTML(html_text))


def get_adv_text(orig_text, used_replacements):
    ''' 
    Get adversarial text from text and list of replacements.
    '''
    text_words = get_tokens(orig_text)
    for (pos, word, replacement_word) in used_replacements:
        assert text_words[pos] == word, 'pos = %d, text_word = %s , word = %s' % (pos, text_words[pos], word)
        text_words[pos] = replacement_word
    return ' '.join(text_words)


class AttackResult:
    '''
    A class for visualizing generated adversarial texts.
    '''
    def __init__(self, results_data):
        self.__dict__ =  results_data


    def get_success_rate(self):
        return sum(self.successes) / len(self.successes)

    def get_modification_percentage(self):
        return sum(self.percents_changed) / len(self.percents_changed)

    def get_success_rate_at_threshold(self,threshold):
        successes = np.array(self.successes)
        percents_changed = np.array(self.percents_changed)
        thresholded_indexes = np.where(np.array(percents_changed) <= threshold)[0]
        return sum(successes[thresholded_indexes]) / len(successes)

    def get_success_rates_at_thresholds(self,thresholds):
        success_rates = dict()
        for threshold in thresholds:
            success_rate_at_threshold = self.get_success_rate_at_threshold(threshold)
            success_rates[threshold] = success_rate_at_threshold
        return success_rates

    def visualize_attack(self, max_display_count = np.inf, show_successful_only = False):
        display_count = min(max_display_count, len(self.sampled_indexes))
        for i in range(display_count):
            print("#%d   index = %d" % (i, self.sampled_indexes[i]))
            if self.successes[i] == 1:
                print("ATTACK SUCCEEDED")
            else :
                print("ATTACK FAILED")
            if show_successful_only and self.successes[i] == 0 :
                print(50*"-")
                continue
            original_prediction = self.original_predictions[i]
            orig_text = self.original_texts[i]
            adversarial_prediction = self.adversarial_predictions[i]
            adv_text = self.adversarial_texts[i]
            orig_html, adv_html = html_render(orig_text, adv_text)
            print("Original prediction: %f" % (original_prediction))
            display_html(orig_html)
            print()
            print("New prediction: %f" % (adversarial_prediction))
            display_html(adv_html)
            print(50*"-")


class FixingResult :
    '''
    A class for visualizing the texts whose classification was corrected.
    '''
    def __init__(self, results_data):
        self.__dict__ =  results_data

    def get_success_rate(self):
        return sum(self.successes) / len(self.successes)

    def get_modification_percentage(self):
        successful_indexes = np.where(np.array(self.successes) == 1)[0]
        return sum(np.array(self.percents_changed)[successful_indexes]) / len(np.array(self.percents_changed)[successful_indexes])

    def get_success_rate_at_threshold(self,threshold):
        successes = np.array(self.successes)
        percents_changed = np.array(self.percents_changed)
        thresholded_indexes = np.where(np.array(percents_changed) <= threshold)[0]
        return sum(successes[thresholded_indexes]) / len(successes)

    def get_success_rates_at_thresholds(self,thresholds):
        success_rates = dict()
        for threshold in thresholds:
            success_rate_at_threshold = self.get_success_rate_at_threshold(threshold)
            success_rates[threshold] = success_rate_at_threshold
        return success_rates

    def visualize(self, max_display_count = np.inf, show_successful_only = False):
        display_count = min(max_display_count, len(self.sampled_indexes))
        for i in range(display_count):
            print("#%d   index = %d" % (i, self.sampled_indexes[i]))
            if self.successes[i] == 1:
                print("FIX SUCCEEDED")
            else :
                print("FIX FAILED: Could not change classification")
            if show_successful_only and self.successes[i] == 0 :
                print(50*"-")
                continue
            original_prediction = self.original_predictions[i]
            orig_text = self.original_texts[i]
            suggestions = [used_replacements for (used_replacements, adversarial_prediction) in self.replacements[i]]
            adversarial_predictions = [adversarial_prediction for (used_replacements, adversarial_prediction) in self.replacements[i]]
            for suggestion_index in range(len(suggestions)):
                print("SUGGESTED REPLACEMENTS OPTION ", suggestion_index, ": ")
                used_replacements = suggestions[suggestion_index]
                adversarial_prediction = adversarial_predictions[suggestion_index]
                adv_text = get_adv_text(orig_text, used_replacements)
                orig_html, adv_html = html_render(orig_text, adv_text)
                print("Original prediction: %f" % (original_prediction))
                display_html(orig_html)
                print("New prediction: %f" % (adversarial_prediction))
                display_html(adv_html)
            print(50*"-")