import re
import numpy as np
import pickle


with open('server/model/vocab_data.pickle', 'rb') as handle:
  vocab_data = pickle.load(handle)

decode_dict = {v:k for k,v in vocab_data['w_idx'].iteritems()}

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]



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
    AA = []
    AB = []
    AC = []
    L = []
    label_num = max([lb[3] for lb in data]) + 1
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
            least_like_q = sorted(ss, cmp = lambda x, y: jaccard(set(x), q_words) < jaccard(set(y), q_words))[:len(ss)- memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in   least_like_q]
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
        AA.append(sa[0])
        AB.append(sa[1])
        AC.append(sa[2])
        L.append(lb)
    Q = pad_sequences(Q, sentence_size)

    return np.array(S), np.array(Q), np.array(AA),np.array(AB),np.array(AC), np.array(L)



def decode(index):
    return decode_dict.get(index, 'unknown')


def process_data(sentences, question):
    sent_t = [tokenize(s.lower()) for s in sentences]
    sent_t = [filter(lambda x: x != ".", s) for s in sent_t]

    q_t = tokenize(question.lower())
    if q_t[-1] == "?":
        q_t = q_t[:-1]

    data = [(sent_t, q_t, ['where'])]

    testS, testQ, testAA, testAB, testAC = vectorize_data(data, vocab_data['w_idx'], vocab_data['sentence_size'], vocab_data['memory_size'])

    return testS, testQ, testAA, testAB, testAC
