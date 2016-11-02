"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

from tensorflow.python.ops import rnn, rnn_cell

def position_encoding(sentence_size, embedding_size):
    J = sentence_size
    d = embedding_size
    l = np.zeros([d,J], dtype=np.float32)
    for k in range(1,d+1):
        for j in range(1, J+1):
            l[k-1,j-1] = (1 - j / float(J)) - (k/float(d))*(1 - 2 * j / float(J))
    return np.transpose(l)


def original_position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope( name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope( name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        answer_size,
        label_size,
        glove_embedding,
        session=tf.Session(),
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        encoding=position_encoding,
        l2 = 0.02,
        lr = 0.01,
        epsilon = 1e-8,
        restoreLoc = None,
        name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._answer_size = answer_size
        self._label_size = label_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._glove_embedding = glove_embedding
        self._opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        self._name = name
        self._l2 = l2
        self._linear = True

        self._rnn_hidden = 50

        self._glove_tf = tf.constant(self._glove_embedding)

        self._init = tf.random_normal_initializer(stddev=0.1)

        self._build_inputs()
        self._build_vars()
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")
        self._answer_encoding = tf.constant(encoding(self._answer_size, self._embedding_size), name="answer_encoding")


        # cross entropy
        logits = self._inference(self._stories, self._queries, self._answers) # (batch_size, label_size)
        print('logits', logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._labels, tf.float32), name="cross_entropy")

        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        #hinge loss
        #hinge_loss = tf.contrib.losses.hinge_loss(logits, tf.cast(self._labels, tf.float32))
        #hinge_loss_sum = tf.reduce_sum(hinge_loss)


        # loss op
        reg_loss = self._l2 * tf.add_n(tf.get_collection('reg_loss'))
        loss_op = cross_entropy_sum + reg_loss
        #loss_op = hinge_loss_sum + reg_loss

        loss_op_summary = tf.scalar_summary("loss", loss_op)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_loss_ema = ema.apply([loss_op])
        loss_ema = ema.average(loss_op)
        self.loss_ema_op = tf.scalar_summary('batch_loss_ema', loss_ema)

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # validation accuracy ops
        self.val_acc_op = self._get_val_acc(predict_op, self._val_labels)
        self.val_acc_summary = tf.scalar_summary("val_acc", self.val_acc_op)

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        self.loss_op_summary = loss_op_summary

        # Summaries
        self.merged = tf.merge_all_summaries()

        self._sess = session

        if restoreLoc is not None:
            saver = tf.train.Saver()
            saver.restore(self._sess, restoreLoc)
        else:
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)


    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._label_size,  self._answer_size], name="answers")
        self._labels = tf.placeholder(tf.int32, [None, self._label_size], name="labels")
        self._val_labels = tf.placeholder(tf.int32, [None], name="val_labels")#TODO: valuation output as index, not as one hot
        self._linear_start= tf.placeholder(tf.bool,[None] , name="linear_start")

    def _build_vars(self):
        with tf.variable_scope(self._name + str(1)):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            self.A = tf.Variable(tf.cast(self._glove_tf, tf.float32), name="A")#TODO: trainable = False
            self.B = tf.Variable(tf.cast(self._glove_tf, tf.float32), name="B")
            self.C = tf.Variable(C, name="C")

            self.TA = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA')
            self.TC = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TC')

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
        self._nil_vars = set([self.A.name, self.B.name, self.C.name])

        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.A))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.B))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.C))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.TA))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.TC))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.H))

    def _inference(self, stories, queries, answers):
        with tf.variable_scope(self._name + str(2)):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            #q_emb = tf.nn.dropout(q_emb, 0.1)
            print('q_emb', q_emb)
            print('self._encoding', self._encoding)
            #u_0 = tf.reduce_sum(q_emb * self._encoding, 1)

            q_step = tf.transpose(q_emb, [1,0,2])
            q_step = tf.reshape(q_step, [-1, self._embedding_size])
            q_step = tf.split(0,self._sentence_size , q_step)
            gru_cell = rnn_cell.GRUCell(self._rnn_hidden)
            outputs, states = rnn.rnn(gru_cell, q_step, dtype=tf.float32)
            print('states', states)
            u_0 = states

            print('u_0', u_0)
            u = [u_0]
            self.probs_hops = []
            for _ in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                #m_emd = tf.nn.dropout(m_emb, 0.1)
                m = tf.reduce_sum(m_emb * self._encoding, 2) + self.TA
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                #probs = tf.nn.softmax(dotted)
                #probs = tf.cond(self._linear_start, lambda: tf.identity(dotted), lambda: tf.nn.softmax(dotted))
                #print('self._linear_start', self._linear_start)
                print('dotted', dotted)
                sfm = tf.nn.softmax(dotted)
                print('sfm', sfm)
                probs = tf.select(self._linear_start, dotted, sfm )
                #print('probs', probs)
                self.probs_hops.append(probs)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

                c_emb = tf.nn.embedding_lookup(self.C, stories)
                c = tf.reduce_sum(c_emb * self._encoding, 2) + self.TC

                c_temp = tf.transpose(c, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                #TRY DROPOUT
                #u_k = tf.nn.dropout(u_k, 0.2)
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            as_emb = tf.nn.embedding_lookup(self.B, answers)
            #as_emb = tf.nn.dropout(as_emb, 0.1)
            print('as_emb', as_emb)
            print('self._answer_encoding', self._answer_encoding)
            as_enc = tf.reduce_sum(as_emb * self._answer_encoding, 2)
            print('as_enc', as_enc)
            print('u_k', u_k)
            u_k_l = tf.tile(u_k, [1, self._label_size])
            u_k_l = tf.reshape(u_k_l, [-1, self._label_size, self._embedding_size])
            print('u_k_l',u_k_l)
            as_ans = tf.sub(as_enc, u_k_l)
            print('as_ans', as_ans)

            as_ans = tf.reduce_sum(tf.square(as_ans), 2)
            print('as_ans', as_ans)
            return as_ans


    def save_model(self, location):
        saver = tf.train.Saver()
        saver.save(self._sess, location, write_meta_graph=False)

    def _get_val_acc(self, pred_op, val_labels):
        corr_pred = tf.equal(tf.cast(pred_op, tf.int32), val_labels)
        acc_op = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
        return acc_op

    def batch_fit(self, stories, queries, answers, labels, linear_start):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        self.linear_start = linear_start
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._labels:labels, self._linear_start:self.linear_start}
        loss, loss_op_summary, _, _, loss_ema, probs = self._sess.run([self.loss_op, self.loss_op_summary, self.train_op, self.update_loss_ema, self.loss_ema_op, self.probs_hops], feed_dict=feed_dict)
        return loss, loss_op_summary, loss_ema, probs

    def predict(self, stories, queries, answers, linear_start):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        self.linear_start = linear_start
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._linear_start:self.linear_start}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_test(self, stories, queries, answers):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers, probabilities per hop: Tensor (None, vocab_size), Tensor (None, hops, memory_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._linear_start:self.linear_start}
        ops = [self.predict_op]
        ops.extend(self.probs_hops)

        return self._sess.run(ops, feed_dict=feed_dict)

    def predict_proba(self, stories, queries, answers):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._linear_start:self.linear_start}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def get_val_acc_summary(self, stories, queries, answers, labels, linear_start):
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._val_labels: labels, self._linear_start:linear_start}
        return self._sess.run([self.val_acc_op, self.val_acc_summary], feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries, answers):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._linear_start:self.linear_start}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)