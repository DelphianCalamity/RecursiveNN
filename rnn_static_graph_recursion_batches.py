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
    max_epochs = 1  # 30
    lr = 0.01
    l2 = 0.02
    batch_size = 3
    max_tree_height = 0

    model_name = MODEL_STR % (embed_size, l2, lr)


class RecursiveNetStaticGraph():

    def __init__(self, config):
        self.config = config

        # Load train data and build vocabulary
        self.train_data, self.dev_data, self.test_data = tree.simplified_data(700,
                                                                              100,
                                                                              200)
        max_height = tree.get_max_tree_height(self.train_data)
        self.config.max_tree_height = pow(2, max_height + 1)
        
        print(self.config.max_tree_height)
        
        # print("data ",self.train_data))
        self.vocab = utils.Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

        # add input placeholders
        dim1 = self.config.batch_size
        dim2 = self.config.max_tree_height

        self.is_leaf_placeholder = tf.placeholder(
            tf.int32, [dim1, dim2], name='is_leaf_placeholder')
        self.node_word_indices_placeholder = tf.placeholder(
            tf.int32, [dim1, dim2], name='node_word_indices_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.int32, [dim1, dim2], name='labels_placeholder')
        self.cons_placeholder = tf.placeholder(
            tf.int32, (None), name='cons')

        # add model variables
        with tf.variable_scope('Embeddings'):
            self.embeddings = tf.get_variable('embeddings',
                                         [len(self.vocab),
                                         self.config.embed_size])

        with tf.variable_scope('Composition'):
            self.W1 = tf.get_variable('W1',
                                 [2 * self.config.embed_size,
                                     self.config.embed_size])
            self.b1 = tf.get_variable('b1', [1, self.config.embed_size]) 

        with tf.variable_scope('Projection'):
            self.U = tf.get_variable('U',
                                [self.config.embed_size,
                                 self.config.label_size])
            self.bs = tf.get_variable('bs', [1, self.config.label_size])


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

            left_node, left_loss = self.rec(i*2, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs)
            right_node, right_loss = self.rec(i*2+1, is_leaf, node_word_indices, embeddings, W, b, labels, U, bs)
            node_tensor = combine_children(left_node, right_node, W, b)
            node_loss = find_loss(node_tensor, i, labels, U, bs)
            loss = tf.concat([left_loss, node_loss, right_loss], 0)

            return [node_tensor, loss]

        # Function Declaration
        self.rec = function.Declare("Rec", [("i", tf.int32), ("is_leaf", tf.int32), ("node_word_indices", tf.int32), 
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


        outloss=[]
        prediction=[]

        for idx_batch in range(self.config.batch_size):

            self.root_prediction, self.full_loss = self.compute_tree(idx_batch)

            prediction.append(self.root_prediction)
            outloss.append(self.full_loss)

        batch_loss = tf.stack(outloss)
        pred = tf.stack(prediction)

        # Compute batch loss
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # regpart = tf.add_n(reg_losses)
        # loss = tf.reduce_mean(batch_loss)
        # self.total_loss = loss + 0.5*regpart
        self.total_loss = tf.reduce_mean(batch_loss)
        # Add training op
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.total_loss)

        # self.writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
        # self.writer.close()

    def compute_tree(self, idx_batch):

        with tf.variable_scope("Composition", reuse=True):

            # Gather is_leaf_placeholder_x
            is_leaf_placeholder_x = tf.gather(self.is_leaf_placeholder, idx_batch)
            # Gather node_word_indices_placeholder_x
            node_word_indices_placeholder_x = tf.gather(self.node_word_indices_placeholder, idx_batch)
            # Gather is_labels_placeholder_x
            labels_placeholder_x = tf.gather(self.labels_placeholder, idx_batch)


            node_tensor, full_loss = self.rec(self.cons_placeholder, is_leaf_placeholder_x, 
                                node_word_indices_placeholder_x, self.embeddings, 
                                self.W1, self.b1, labels_placeholder_x, self.U, self.bs)

           # add projection layer
            root_logits = tf.matmul(node_tensor, self.U) + self.bs
            root_prediction = tf.squeeze(tf.argmax(root_logits, 1))
          
            # add loss layer
            root_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=root_logits, labels=labels_placeholder_x[1:2]))
            regularization_loss = self.config.l2 * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.U))
            full_loss = regularization_loss + tf.reduce_sum(full_loss)

        return root_prediction, full_loss


    def build_feed_dict(self, batch_data):

        # size = pow(2, atree.max_depth + 1)
        # size = pow(2, self.config.max_tree_height + 1)
 
        def assign(node, args):
            args[0][args[1]] = node
 
        is_leaf = []
        node_word_indices = []
        labels = []
        # print(self.config.max_tree_height)
        for _, atree in enumerate(batch_data):

            nodes_list = [None] * self.config.max_tree_height
            root = atree.root
            tree.traverse(root, assign, (nodes_list, 1))

            is_leaf.append([0 if node is None or not node.isLeaf else 1 for node in nodes_list])
            node_word_indices.append([self.vocab.encode(node.word) if node is not None and node.word else -1 for node in nodes_list])
            labels.append([node.label if node is not None else -1 for node in nodes_list])

        # print(labels)
        feed_dict = {
            self.cons_placeholder: 1,
            # Using int32 instead of bool for by-passing errors occuring during the construction of the function's gradient
            self.is_leaf_placeholder: is_leaf,
            self.node_word_indices_placeholder: node_word_indices,
            self.labels_placeholder: labels
        }
        return feed_dict


    def run_epoch(self, sess=None, new_model=False, verbose=True):
        loss_history = []
        # training
        # random.shuffle(self.train_data)
        if new_model:
            sess.run(tf.initialize_all_variables())

        data_idxs = list(range(len(self.train_data)))
        # shuffle(data_idxs)
        # step=0
        for i in range(0, len(self.train_data), self.config.batch_size):
            # 0, 5, 10, 15, .. len(data)
            batch_size = min(i+self.config.batch_size, len(self.train_data))-i
            if batch_size < self.config.batch_size: break

            batch_idxs = data_idxs[i:i+self.config.batch_size]
            batch_data = [self.train_data[ix] for ix in batch_idxs]

            feed_dict = self.build_feed_dict(batch_data)
           
            loss_value, _ = sess.run([self.total_loss, self.train_op], feed_dict=feed_dict)
            loss_history.append(loss_value)
           
            # step = step+1
            # if verbose:
            #     sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
            #         self.train_data), np.mean(loss_history)))
            #     sys.stdout.flush()
                

        # statistics
        # Root Prediction
        # train_preds = []

        # for tree in self.train_data:
        #     feed_dict = self.build_feed_dict(tree)
        #     root_prediction = sess.run(
        #                 self.root_prediction, feed_dict=feed_dict)
        #     train_preds.append(root_prediction)

        # val_preds = []
        # val_losses = []
        # for tree in self.dev_data:
        #     feed_dict = self.build_feed_dict(tree)
        #     root_prediction, loss = sess.run(
        #             [self.root_prediction, self.root_loss], feed_dict=feed_dict)
        #     val_losses.append(loss)
        #     val_preds.append(root_prediction)

        # train_labels = [t.root.label for t in self.train_data]
        # val_labels = [t.root.label for t in self.dev_data]
        # train_acc = np.equal(train_preds, train_labels).mean()
        # val_acc = np.equal(val_preds, val_labels).mean()
        
        # print('\nTraining acc (only root node): {}'.format(train_acc))
        # print('Valiation acc (only root node): {}'.format(val_acc))
        # print(self.make_conf(train_labels, train_preds))
        # print(self.make_conf(val_labels, val_preds))
        
        # return train_acc, val_acc, loss_history, np.mean(val_losses)
        return None, None, loss_history, None

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
            # complete_loss_history.extend(loss_history)
            # train_acc_history.append(train_acc)
            # val_acc_history.append(val_acc)
            # # lr annealing
            # epoch_loss = np.mean(loss_history)
            # if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
            #     self.config.lr /= self.config.anneal_by
            #     print('annealed lr to %f' % self.config.lr)
            # prev_epoch_loss = epoch_loss

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_val_epoch = epoch

            # # if model has not improved for a while stop
            # if epoch - best_val_epoch > self.config.early_stopping:
            #     stopped = epoch

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

        sess = tf.Session()

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

        sess = tf.Session()

        start_time = time.time()
        stats = self.train(verbose=True, sess=sess)
        print('\nTraining time: {}'.format(time.time() - start_time))

        self.plot_loss_history(stats)


if __name__ == '__main__':

    config = Config()
    model = RecursiveNetStaticGraph(config)
    # model.test_RNN(config)
    model.train_RNN(config)