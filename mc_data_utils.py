from __future__ import absolute_import

import os
import re
import numpy as np


def load_task(data_dir, task_id):
    '''Load readworks tasks

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id in [1,2]

    train_file = os.path.join(data_dir, 'readworks_grade{}.0.1.json'.format(task_id))
    test_file = os.path.join(data_dir, 'readworks_grade{}.dev.0.1.json'.format(task_id))
    train_data = json_get_data(train_file)
    test_data = json_get_data(test_file)
    return train_data, test_data

import json
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def my_sent_tokenize(txt):
    tmp = sent_tokenize(txt)
    ret = []
    for t1 in tmp:
      if t1[0].islower():
        ret[len(ret)-1] = ret[len(ret)-1] + ' ' + t1
      else:
        ret.append(t1)
    return ret

def sent_to_tokens(sent):
    return [lemmatizer.lemmatize(tmp) for tmp in word_tokenize(sent.lower())]

def para_to_tokens(para):
    return [sent_to_tokens(sent) for sent in my_sent_tokenize(para)]


def json_get_data(fname):
    with open(fname) as f:
      lines = f.read()
    jlines = json.loads(lines)
    ret = []
    for exjson in jlines['exercises']:
      s_tokens = para_to_tokens(exjson['story']['text'])
      for qjson in exjson['questions']:
        q_tokens = sent_to_tokens(qjson['text'])
        a_dict = {'A':'','B':'','C':''}
        for ansjson in qjson['answerChoices']:
          a_dict[ansjson['label']] = ansjson['text']
        a_tokens = []
        for lab in 'ABC':
          a_tokens.append(sent_to_tokens(a_dict[lab]))
        l_dict = {'A':0,'B':1,'C':2}
        true_ans = l_dict[qjson['correctAnswer']]
        ret.append([s_tokens, q_tokens, a_tokens, true_ans])
    return ret


def get_vocab(data):
    assert(len(data[0]) == 4)
    vocab = set()
    ret = [u'<eos>']
    for dp in data:
        vocab = vocab.union(set(chain.from_iterable(dp[0])))
        vocab = vocab.union(set(dp[1]))
        vocab = vocab.union(set(chain.from_iterable(dp[2])))
    ret.extend(sorted(list(vocab)))
    return ret


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data



def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        if len(ss) > memory_size:
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = set(q)
            least_like_q = sorted(ss, cmp = lambda x, y: jaccard(set(x), q_words) < jaccard(set(y), q_words))[:len(ss)-memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                ss.remove(sent)
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)

def jaccard(a, b):
    '''
    Assumes that a and b are sets so that calling code only has to cast the question to set once.
    '''
    return len(a.intersection(b)) / float(len(a.union(b)))
    set(a).intersection(set(b))
