from mc_memn2n import MemN2N
import tensorflow as tf
import numpy as np
import os

import pickle
"""
Stores the configuration for loading the model
max_memory_size is the maximum size allowed during training
memory_size is the actual memory size used based on max story size
"""



import cPickle
def load_glove(dim):
    """ Loads GloVe data.
    :param dim: word vector size (50, 100, 200)
    :return: GloVe word table
    """
    word2vec = {}
    print('start loading glove')

    path = "../data/glove/glove.6B." + str(dim) + 'd'
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
    fn = "../data/glove/myembedding." + str(dim) + "d"+ '.pickle'
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




with open('../save/vocab_data.pickle', 'rb') as handle:
  vocab_data = pickle.load(handle)

answer_size  = 60
vocab_size = 7023
embedding_size = 50
batch_size = 64
memory_size = 100
label_size = 4
l2 = 0.1
restoreLoc = '../save/weights/myweights'
sentence_size = vocab_data['sentence_size']
glove_embedding = get_embedding(vocab_data['vocab'], embedding_size, 1)



sess = tf.Session()


model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size, answer_size, label_size,   glove_embedding = glove_embedding,session=sess, l2=l2, nonlin=tf.nn.relu, restoreLoc=restoreLoc)

print('finished load model')

# Uncomment to see if the weights were loaded correctly
# print(sess.run(model.A))

def get_pred(testS, testQ, testAS):
    ps = model.predict_proba(testS, testQ, testAS)
    op = model.predict_test(testS, testQ, testAS)

    answer = op[0][0]
    print('answer', answer)
    answer_probability = float(np.max(ps))
    print('answer_probability', answer_probability)
    mem_probs = np.vstack(op[1:]).T[testS[0].any(axis=1)]
    return answer, answer_probability, mem_probs




