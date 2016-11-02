import re
import numpy as np
import pickle


with open('../save/vocab_data.pickle', 'rb') as handle:
  vocab_data = pickle.load(handle)

decode_dict = {v:k for k,v in vocab_data['w_idx'].iteritems()}

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


from mc_data_utils import vectorize_data, sent_to_tokens, para_to_tokens

def decode(index):
    return decode_dict.get(index, 'unknown')


def process_data(sentences, question, choices):
    print('sentences', sentences)
    sent_t = [sent_to_tokens(tmp) for tmp in sentences]
    print('sent_t', sent_t)
    #sent_t = [filter(lambda x: x != ".", s) for s in sent_t]

    q_t = sent_to_tokens(question)
    #if q_t[-1] == "?":
    #    q_t = q_t[:-1]

    choices_t = choices.split('|')
    c_t = [sent_to_tokens(tmp) for tmp in choices_t]

    data = [(sent_t, q_t, c_t,'0' )]

    testS, testQ, testAS, testL = vectorize_data(data, vocab_data['w_idx'], vocab_data['sentence_size'], vocab_data['memory_size'], vocab_data['answer_size'])

    return testS, testQ, testAS, testL
