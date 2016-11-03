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


config = {
    'batch': 64,
    'vocab_size': len(vocab_data['vocab']),
    'sentence_size': vocab_data['sentence_size'],
    'max_memory_size': vocab_data['memory_size'],
    'memory_size': vocab_data['memory_size'],
    'embedding_size': 50,
    'hops': 3,
    'max_grad_norm': 40.0,
    'regularization': 0.1,
    'epsilon': 1e-8,
    'lr': 0.001
}

restore_location = '../save/weights/wts_pe'
print(restore_location)

sess = tf.Session()

model = MemN2N(batch_size=config["batch"],
               vocab_size=config["vocab_size"],
               sentence_size=config["sentence_size"],
               memory_size=config["memory_size"],
               embedding_size=config["embedding_size"],
               answer_size=vocab_data['answer_size'],
               label_size=4,
               glove_embedding=np.random.standard_normal([config['vocab_size'], config['embedding_size']]),
               session=sess,
               hops=config["hops"],
               max_grad_norm=config["max_grad_norm"],
               l2=config["regularization"],
               lr=config["lr"],
               epsilon=config["epsilon"],
               nonlin=tf.nn.relu,
               restoreLoc=restore_location)

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
