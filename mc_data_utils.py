from __future__ import absolute_import

import os
import re
import numpy as np
from itertools import chain


def load_task(data_dir, task_id):
    '''Load readworks tasks

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id in [1,2]
    #task_id= 1
    #data_dir = './data/readworksAll/'
    #train_file = os.path.join(data_dir, 'readworks_grade{}.0.1.json'.format(task_id))
    #test_file = os.path.join(data_dir, 'readworks_grade{}.0.1.json'.format(task_id))
    #train_data = json_get_data(train_file)
    data_dir = './data/readworksTrainTest2/'
    train_file = os.path.join(data_dir, 'readworks_grade{}.test.0.1.json'.format(task_id))
    fns = ['readworks_grade1.test.0.1.json','readworks_grade2.test.0.1.json','readworks_grade2.dev.0.1.json']
    test_file = os.path.join(data_dir, 'readworks_grade{}.dev.0.1.json'.format(task_id))
    train_data = []
    for fn in fns:
        fn_path = os.path.join(data_dir, fn)
        td = json_get_data(fn_path)
        train_data.extend(td)
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
      if 'text' not in exjson['story']:
        print("No 'text' in story")
        print(exjson)
        continue
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
        if qjson['correctAnswer'] not in ['A','B','C']:
            print("CorrectAnswer not in 'ABC'")
            print(qjson)
            continue
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


from keras.preprocessing.sequence import pad_sequences
#TODO: for some reason just converting Q into np.array doesn't convert into a multidimensional array. Use pad_sequences from keras to reinforce.


def vectorize_data(data, word_idx, sentence_size, memory_size,answer_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    AS = []
    L = []
    label_nums = set([len(dt[2]) for dt in data])
    assert(len(label_sums) == 1)
    label_num = list(label_nums)[0]
    for story, query, answer, label in data:
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

        sa = []
        for i, sentence in enumerate(answer, 1):
            ls = max(0, answer_size - len(sentence))
            sa.append([word_idx[w] for w in sentence] + [0] * ls)

        lb = np.zeros(label_num)
        lb[label] = 1

        S.append(ss)
        Q.append(q)
        AS.append(sa)
        L.append(lb)
    Q = pad_sequences(Q, sentence_size)

    return np.array(S), np.array(Q), np.array(AS), np.array(L)


def perturb(S,Q,AA,AB,AC,L):
    """deprecated"""
    retS = S
    retQ = Q
    retAA = AA
    retAB = AB
    retAC = AC
    retL = L

    As = [AA,AB,AC]

    for vec in [[0,2,1],[1,0,2],[1,2,0],[2,1,0],[2,0,1]]:
        retS = np.vstack([retS,S])
        retQ = np.vstack([retQ,Q])
        retAA = np.vstack([retAA, As[vec[0]]])
        retAB = np.vstack([retAB, As[vec[1]]])
        retAC = np.vstack([retAC, As[vec[2]]])
        retL = np.vstack([retL, L[:,vec]])
    print("number of data points after perturbation", len(retS))
    print("number of data points after perturbation", len(retQ))
    print("number of data points after perturbation", len(retAA))
    print("number of data points after perturbation", len(retAB))
    print("number of data points after perturbation", len(retAC))
    print("number of data points after perturbation", len(retL))
    idxs = np.arange(0, len(retS))
    np.random.shuffle(idxs)
    retS = retS[idxs, :]
    retQ = retQ[idxs, :]
    retAA = retAA[idxs, :]
    retAB = retAB[idxs, :]
    retAC = retAC[idxs, :]
    retL = retL[idxs, :]
    return retS, retQ, retAA, retAB, retAC, retL





def jaccard(a, b):
    '''
    Assumes that a and b are sets so that calling code only has to cast the question to set once.
    '''
    return len(a.intersection(b)) / float(len(a.union(b)))
    set(a).intersection(set(b))

import cPickle
def load_glove(dim):
    """ Loads GloVe data.
    :param dim: word vector size (50, 100, 200)
    :return: GloVe word table
    """
    word2vec = {}
    print('start loading glove')

    path = "data/glove/glove.6B." + str(dim) + 'd'
    fn = path+ '.cache'
    if os.path.exists(fn) and os.stat(fn).st_size > 0:
        with open(fn, 'rb') as cache_file:
            word2vec = cPickle.load(cache_file)

    else:
        # Load n create cache
        with open(path + '.txt') as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = [float(x) for x in l[1:]]

        with open(path + '.cache', 'wb') as cache_file:
            cPickle.dump(word2vec, cache_file)

    print("Loaded Glove data")
    return word2vec

def get_embedding(vocab, dim=50, use_pickle=1):
    if dim not in [50,100,200,300]:
        print("Not using glove")
        return np.random.standard_normal([len(vocab),dim])
    print('use pickle', use_pickle)
    fn = "data/glove/myembedding." + str(dim) + "d"+ '.pickle'
    if os.path.exists(fn) and os.stat(fn).st_size > 0 and use_pickle == 1:
        with open(fn, 'rb') as pickle_file:
            ret = cPickle.load(pickle_file)
    else:
        wv = load_glove(dim)
        ret = np.zeros([len(vocab), dim])
        for ii in range(len(vocab)):
            if ii == 0:
                continue
            else:
                if vocab[ii] in wv:
                    ret[ii,:] = wv[vocab[ii]]
                else:
                    ret[ii,:] = np.random.randn(dim) * 0.5

        with open(fn, 'wb') as pickle_write:
            cPickle.dump(ret, pickle_write)
    return ret






