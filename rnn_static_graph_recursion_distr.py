import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils
import tree

from tensorflow.python.framework import function

MODEL_STR = 'rnn_embed=%d_l2=%f_lr=%f.weights'
SAVE_DIR = './weights/'


class Config(object):
    """Holds model hyperparams and data information.
    Model objects are passed a Config() object at instantiation.
    """
    embed_size = 35
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 5  # 30
    lr = 0.01
    l2 = 0.02

    model_name = MODEL_STR % (embed_size, l2, lr)


class RecursiveNetStaticGraph():

    def __init__(self, config):
        self.config = config

        # Load train data and build vocabulary
        self.train_data, self.dev_data, self.test_data = tree.simplified_data(700,
                                                                              100,
                                                                              200)
        # print("data ",self.train_data))
        self.vocab = utils.Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

        cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

        # add input placeholders
        self.is_leaf_placeholder = tf.placeholder(
            tf.int32, (None), name='is_leaf_placeholder')
        self.node_word_indices_placeholder = tf.placeholder(
            tf.int32, (None), name='node_word_indices_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.int32, (None), name='labels_placeholder')
        self.cons_placeholder = tf.placeholder(
            tf.int32, (None), name='cons')

        # add model variables
        # making initialization deterministic for now
        # initializer = tf.random_normal_initializer(seed=1)
        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings',
                                         [len(self.vocab),
                                         self.config.embed_size])

        with tf.variable_scope('Composition'):
            W1 = tf.get_variable('W1',
                                 [2 * self.config.embed_size,
                                     self.config.embed_size])
            b1 = tf.get_variable('b1', [1, self.config.embed_size]) 

        with tf.variable_scope('Projection'):
            U = tf.get_variable('U',
                                [self.config.embed_size,
                                 self.config.label_size])
            bs = tf.get_variable('bs', [1, self.config.label_size])

        # Build recursive graph
        def embed_word(word_index, embeddings):
            return tf.expand_dims(tf.gather(embeddings, word_index), 0)

        def combine_children(left_tensor, right_tensor, W, b):
            return tf.nn.relu(tf.matmul(tf.concat([left_tensor, right_tensor], 1), W) + b)

        def find_loss(node_tensor, i, labels, U, bs):
            # add projection layer
            node_logits = tf.matmul(node_tensor, U) + bs
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=node_logits, labels=labels[i:i+1])
            return loss

        def base_case(node_word_indices, i, embeddings, labels, U, bs):

            word_index = tf.gather(node_word_indices, i)
            node_tensor = embed_word(word_index, embeddings)
            loss = find_loss(node_tensor, i, labels, U, bs)
            
            return [node_tensor, loss]

        def rec_case(i, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs):

            with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
                left_node, left_loss = rec(i*2, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs)
                right_node, right_loss = rec(i*2+1, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs)

            with tf.device("/job:local/replica:0/task:1/device:CPU:0"):        
                node_tensor = combine_children(left_node, right_node, W, b)

            node_loss = find_loss(node_tensor, i, labels, U, bs)
            loss = tf.concat([left_loss, node_loss, right_loss], 0)

            return [node_tensor, loss]


        # Function Declaration
        rec = function.Declare("Rec", [("i", tf.int32), ("is_leaf", tf.int32), ("node_word_indices", tf.int32), 
            ("embeddings", tf.float32), ("W", tf.float32), ("b", tf.float32), ("labels", tf.int32), ("U", tf.float32), ("bs", tf.float32)], 
            [("ret", tf.float32), ("ret1", tf.float32)])

        # Function Definition
        @function.Defun(tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, func_name="Rec", grad_func="GradFac", create_grad_func=True, out_names=["ret", "ret1"])
        def RecImpl(i, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs):
            node_tensor, loss = \
                tf.cond(tf.equal(tf.gather(is_leaf, i), tf.constant(1)),
                        lambda: base_case(node_word_indices, i, embeddings, labels, U, bs),
                        lambda: rec_case(i, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs))
            return [node_tensor, loss]

        RecImpl.add_to_graph(tf.get_default_graph())


        self.node_tensor, self.full_loss = rec(self.cons_placeholder, self.is_leaf_placeholder, 
                            self.node_word_indices_placeholder, self.embeddings, W1, b1, self.labels_placeholder, U, bs)

        # add projection layer
        self.root_logits = tf.matmul(self.node_tensor, U) + bs
        self.root_prediction = tf.squeeze(tf.argmax(self.root_logits, 1))
      
        # add loss layer
        with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
            l1 = tf.nn.l2_loss(W1)
        with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
            l2 = tf.nn.l2_loss(U)

        l = l1 + l2
        regularization_loss = self.config.l2*l

        with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
            x = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.root_logits, labels=self.labels_placeholder[1:2])
        self.root_loss = regularization_loss + tf.reduce_sum(x)
        
        # # add training op
        self.full_loss = tf.reduce_sum(self.full_loss)
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.full_loss)

        # self.writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
        # self.writer.close()


    def build_feed_dict(self, atree):
        size = pow(2, atree.max_depth + 1)
        nodes_list = [None] * size
        root = atree.root

        def assign(node, args):
            args[0][args[1]] = node

        tree.traverse(root, assign, (nodes_list, 1))

        feed_dict = {
            self.cons_placeholder: 1,

            # Using int32 instead of bool for by-passing errors occuring during the construction of the function's gradient
            self.is_leaf_placeholder: [0 if 
                                      node is None or not node.isLeaf else 1 
                                      for node in nodes_list],

            self.node_word_indices_placeholder: [self.vocab.encode(node.word) if
                                                 node is not None and node.word else -1
                                                 for node in nodes_list],

            self.labels_placeholder: [node.label if
                                      node is not None else -1
                                      for node in nodes_list]
        }
        return feed_dict

    def run_epoch(self, sess=None, new_model=False, verbose=True):
        loss_history = []
        # training
        # random.shuffle(self.train_data)
        if new_model:
            sess.run(tf.initialize_all_variables())

        for step, tree in enumerate(self.train_data):
            feed_dict = self.build_feed_dict(tree)
            # print(self.embeddings.eval(session=sess))
            loss_value, _ = sess.run([self.full_loss, self.train_op], feed_dict=feed_dict)
            # print(loss_value)
            # print(self.embeddings.eval(session=sess))
            loss_history.append(loss_value)
            if verbose:
                sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
                    self.train_data), np.mean(loss_history)))
                sys.stdout.flush()
                
        # statistics

        # Root Prediction
        train_preds = []
        for tree in self.train_data:
            feed_dict = self.build_feed_dict(tree)
            root_prediction = sess.run(
                        self.root_prediction, feed_dict=feed_dict)
            train_preds.append(root_prediction)

        val_preds = []
        val_losses = []
        for tree in self.dev_data:
            feed_dict = self.build_feed_dict(tree)
            root_prediction, loss = sess.run(
                    [self.root_prediction, self.root_loss], feed_dict=feed_dict)
            val_losses.append(loss)
            val_preds.append(root_prediction)

        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()
        
        print('\nTraining acc (only root node): {}'.format(train_acc))
        print('Valiation acc (only root node): {}'.format(val_acc))
        print(self.make_conf(train_labels, train_preds))
        print(self.make_conf(val_labels, val_preds))
        
        return train_acc, val_acc, loss_history, np.mean(val_losses)

    def train(self, sess=None, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in range(self.config.max_epochs):
            print('epoch %d' % epoch)
            if epoch == 0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(sess=sess,
                    new_model=True)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(sess=sess)
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            # lr annealing
            epoch_loss = np.mean(loss_history)
            if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
                self.config.lr /= self.config.anneal_by
                print('annealed lr to %f' % self.config.lr)
            prev_epoch_loss = epoch_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

            # if model has not improved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch

        if verbose:
          sys.stdout.write('\r')
          sys.stdout.flush()

        print('\n\nstopped at %d\n' % stopped)
        return {
            'loss_history': complete_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l, p in zip(labels, predictions):
            confmat[l, p] += 1
        return confmat

    def plot_loss_history(self, stats):
        plt.plot(stats['loss_history'])
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('loss_history.png')
        plt.show()

    def test_RNN(self, config):
        """Test RNN model implementation. """
        # graph_def = tf.get_default_graph().as_graph_def()
        # with open('static_graph.pb', 'wb') as f:
        #  f.write(graph_def.SerializeToString())

        sess = tf.Session("grpc://localhost:2222")

        start_time = time.time()
        stats = self.train(verbose=True, sess=sess)
        print('\nTraining time: {}'.format(time.time() - start_time))

        self.plot_loss_history(stats)
        
        start_time = time.time()

        val_preds = []
        val_losses = []
        for tree in self.dev_data:
            feed_dict = self.build_feed_dict(tree)
            root_prediction, loss = sess.run(
                    [self.root_prediction, self.root_loss], feed_dict=feed_dict)
            val_losses.append(loss)
            val_preds.append(root_prediction)

        val_labels = [t.root.label for t in self.dev_data]
        val_acc = np.equal(val_preds, val_labels).mean()
        print(val_acc)
        
        print('-' * 20)
        print('Test')

        predictions = []
        for tree in self.test_data:
            feed_dict = self.build_feed_dict(tree)
            root_prediction = sess.run(
                        self.root_prediction, feed_dict=feed_dict)
            predictions.append(root_prediction)

        labels = [t.root.label for t in self.test_data]
        print(self.make_conf(labels, predictions))
        test_acc = np.equal(predictions, labels).mean()
        print('Test acc: {}'.format(test_acc))
        print('Inference time, dev+test: {}'.format(time.time() - start_time))
        print('-' * 20)

        sess.close()

    def train_RNN(self, config):

        sess = tf.Session("grpc://localhost:2222")

        start_time = time.time()
        stats = self.train(verbose=True, sess=sess)
        print('\nTraining time: {}'.format(time.time() - start_time))

        self.plot_loss_history(stats)


if __name__ == '__main__':

    config = Config()
    model = RecursiveNetStaticGraph(config)
    # model.test_RNN(config)
    model.train_RNN(config)