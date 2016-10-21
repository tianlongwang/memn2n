"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from mc_data_utils import load_task, vectorize_data, get_vocab, get_embedding, perturb
from sklearn import cross_validation, metrics
from memn2n.mc_memn2n import MemN2N
from itertools import chain
from six.moves import range


import os
import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("regularization", 0.2, "Regularization.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 50, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/readworksAll/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("cache_embedding", 1, "Use embedding cache. If new data from previous one, do not use")
FLAGS = tf.flags.FLAGS

def get_log_dir_name():
    lr = FLAGS.learning_rate
    eps = FLAGS.epsilon
    mgn = FLAGS.max_grad_norm
    hp = FLAGS.hops
    es = FLAGS.embedding_size
    ms = FLAGS.memory_size
    ti = FLAGS.task_id
    reg = FLAGS.regularization

    log_dir_name = "lr={0}_eps={1}_mgn={2}_hp={3}_es={4}_ms={5}_reg={6}".format(lr, eps, mgn, hp, es, ms, reg)
    return os.path.join('./logs', str(ti), log_dir_name)

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = get_vocab(data)

word_idx = dict((c, i ) for i, c in enumerate(vocab))


glove_embedding = get_embedding(vocab, FLAGS.embedding_size, FLAGS.cache_embedding)



max_story_size = max(map(len, (s for s, _, _ ,_ in data)))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _,_ in data)))
query_size = max(map(len, (q for _, q, _,_ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word #TODO: already got <eos>??
sentence_size = max(query_size, sentence_size) # for the position
answer_size = max(map(len, chain.from_iterable(a for _,_,a,_ in data)))
label_size = len(data[0][2])

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Answer length", answer_size)
print("Label size", label_size)

# train/validation/test sets
S, Q, A, L = vectorize_data(train, word_idx, sentence_size, memory_size, answer_size)
trainS, valS, trainQ, valQ, trainA, valA, trainL, valL = cross_validation.train_test_split(S, Q, A, L, test_size=.2, random_state=FLAGS.random_state)
#trainS,trainQ,trainAA,trainAB,trainAC,trainL = perturb(trainS,trainQ,trainAA,trainAB,trainAC,trainL)
testS, testQ, testA, testL= vectorize_data(test, word_idx, sentence_size, memory_size, answer_size)

#print(testS[0])

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainL, axis=1)
test_labels = np.argmax(testL, axis=1)
val_labels = np.argmax(valL, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size


batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, answer_size, label_size, glove_embedding = glove_embedding,session=sess,
                     l2=FLAGS.regularization, nonlin=tf.nn.relu)

    writer = tf.train.SummaryWriter(get_log_dir_name(), sess.graph)

    for t in range(1, FLAGS.epochs+1):
        np.random.shuffle(batches)
        total_cost = 0.0
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            l = trainL[start:end]
            cost_t, cost_summary, cost_ema = model.batch_fit(s, q, a, l)
            total_cost += cost_t

            # writer.add_summary(cost_summary, t*n_train+start)
            writer.add_summary(cost_ema, t*n_train+start)

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                pred = model.predict(s, q, a)
                train_preds += list(pred)

#             val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
            total_cost_summary = tf.scalar_summary("epoch_loss", total_cost)
            tcs = sess.run(total_cost_summary)
            writer.add_summary(tcs, t)
#             val_acc = metrics.accuracy_score(val_preds, val_labels)

            val_acc, val_acc_summary = model.get_val_acc_summary(valS, valQ, valA, val_labels)
            writer.add_summary(val_acc_summary, t)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

            test_preds = model.predict(testS, testQ, testA)
            test_acc = metrics.accuracy_score(test_preds, test_labels)
            print("Testing Accuracy:", test_acc)
