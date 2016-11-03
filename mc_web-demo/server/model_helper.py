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
