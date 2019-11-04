import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils
import tree

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
  max_epochs = 5 # 30
  lr = 0.01
  l2 = 0.02
  batch_size = 5
  max_tree_nodes = 0

  model_name = MODEL_STR % (embed_size, l2, lr)


class RecursiveNetStaticGraph():

  def __init__(self, config):
    self.config = config

    # Load train data and build vocabulary
    self.train_data, self.dev_data, self.test_data = tree.simplified_data(700,
                                                                          100,
                                                                          200)
    self.config.max_tree_nodes = tree.get_max_tree_nodes(self.train_data + self.dev_data + self.test_data)
        
    print(self.config.max_tree_nodes)

    # print("data ",self.train_data))
    self.vocab = utils.Vocab()
    train_sents = [t.get_words() for t in self.train_data]
    self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    # add input placeholders
    dim1 = self.config.batch_size
    dim2 = self.config.max_tree_nodes

    self.is_leaf_placeholder = tf.placeholder(
        tf.bool, [dim1, dim2], name='is_leaf_placeholder')
    self.left_children_placeholder = tf.placeholder(
        tf.int32, [dim1, dim2], name='left_children_placeholder')
    self.right_children_placeholder = tf.placeholder(
        tf.int32, [dim1, dim2], name='right_children_placeholder')
    self.node_word_indices_placeholder = tf.placeholder(
        tf.int32, [dim1, dim2], name='node_word_indices_placeholder')
    self.labels_placeholder = tf.placeholder(
        tf.int32, [dim1, dim2], name='labels_placeholder')
    self.tree_size_placeholder = tf.placeholder(
        tf.int32, [dim1], name='tree_size_placeholder')
    # add model variables
    # making initialization deterministic for now
    # initializer = tf.random_normal_initializer(seed=1)
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

    outloss = []
    prediction = []
    root_loss = []

    for idx_batch in range(self.config.batch_size):

        self.root_prediction, self.full_loss, self.root_loss = self.compute_tree(idx_batch)

        prediction.append(self.root_prediction)
        outloss.append(self.full_loss)
        root_loss.append(self.root_loss)

    batch_loss = tf.stack(outloss)
    self.pred = tf.stack(prediction)
    self.rloss = tf.stack(root_loss)

    # Compute batch loss
    self.total_loss = tf.reduce_mean(batch_loss)
    # Add training op
    self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.total_loss)


  def compute_tree(self, idx_batch):

    tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

    def embed_word(word_index):
      # with tf.device('/cpu:0'):
        return tf.expand_dims(tf.gather(self.embeddings, word_index), 0)

    def combine_children(left_tensor, right_tensor):
      return tf.nn.relu(tf.matmul(tf.concat([left_tensor, right_tensor],1), self.W1) + self.b1)

    def loop_body(tensor_array, i):
      node_is_leaf = tf.gather(self.is_leaf_placeholder_x, i)
      node_word_index = tf.gather(self.node_word_indices_placeholder_x, i)
      left_child = tf.gather(self.left_children_placeholder_x, i)
      right_child = tf.gather(self.right_children_placeholder_x, i)
      print(left_child, "left_child")
      node_tensor = tf.cond(
          node_is_leaf,
          lambda: embed_word(node_word_index),
          lambda: combine_children(tensor_array.read(left_child),
                                   tensor_array.read(right_child)))
      tensor_array = tensor_array.write(i, node_tensor)
      i = tf.add(i, 1)
      return tensor_array, i

    self.is_leaf_placeholder_x = tf.gather(self.is_leaf_placeholder, idx_batch)
    self.node_word_indices_placeholder_x = tf.gather(self.node_word_indices_placeholder, idx_batch)
    self.labels_placeholder_x = tf.gather(self.labels_placeholder, idx_batch)
    self.left_children_placeholder_x = tf.gather(self.left_children_placeholder, idx_batch)
    self.right_children_placeholder_x = tf.gather(self.right_children_placeholder, idx_batch)
    self.tree_size_placeholder_x = tf.gather(self.tree_size_placeholder, idx_batch)

    loop_cond = lambda tensor_array, i: \
        tf.less(i, self.tree_size_placeholder_x) #tf.squeeze(tf.shape(self.is_leaf_placeholder_x)))
    self.tensor_array, _ = tf.while_loop(loop_cond, loop_body, [tensor_array, 0], parallel_iterations=1)

    # add projection layer
    logits = tf.matmul(self.tensor_array.concat(), self.U) + self.bs
    root_logits = tf.matmul(self.tensor_array.read(self.tensor_array.size() - 1), self.U) + self.bs
    root_prediction = tf.squeeze(tf.argmax(root_logits, 1))

    # add loss layer
    regularization_loss = self.config.l2 * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.U))
    # included_indices = tf.where(tf.less(self.labels_placeholder, 2))
    full_loss = regularization_loss + tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder_x[0:self.tree_size_placeholder_x]))

    root_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=root_logits,labels=self.labels_placeholder_x[self.tree_size_placeholder_x-1:self.tree_size_placeholder_x]))

    return root_prediction, full_loss, root_loss

  def build_feed_dict(self, batch_data):

    is_leaf = []
    left_children = []
    right_children = []
    node_word_indices = []
    labels = []
    tree_size = []

    for _, atree in enumerate(batch_data):

      nodes_list = []
      tree.leftTraverse(atree.root, lambda node, args: args.append(node), nodes_list)
      node_to_index = OrderedDict()
      for i in range(len(nodes_list)):
        node_to_index[nodes_list[i]] = i

      nodes_list += (self.config.max_tree_nodes-len(nodes_list))*[None]

      is_leaf.append([False if node is None else node.isLeaf for node in nodes_list])
      left_children.append([-1 if node is None or node.isLeaf else node_to_index[node.left]  for node in nodes_list])
      right_children.append([-1 if node is None or node.isLeaf else node_to_index[node.right]  for node in nodes_list])
      node_word_indices.append([-1 if node is None or not node.word else self.vocab.encode(node.word) for node in nodes_list])
      labels.append([-1 if node is None else node.label for node in nodes_list])
      tree_size.append(atree.num_nodes)

    feed_dict = {
        self.is_leaf_placeholder: is_leaf,
        self.left_children_placeholder: left_children, 
        self.right_children_placeholder: right_children,
        self.node_word_indices_placeholder: node_word_indices,
        self.labels_placeholder: labels,
        self.tree_size_placeholder: tree_size
    }
    return feed_dict

  def run_epoch(self, sess=None, new_model=False, verbose=True):
    loss_history = []
    # training
    # random.shuffle(self.train_data)
    if new_model:
        sess.run(tf.initialize_all_variables())

    data_idxs = list(range(len(self.train_data)))
    step=0
    for i in range(0, len(self.train_data), self.config.batch_size):
      # 0, 5, 10, 15, .. len(data)
      batch_size = min(i+self.config.batch_size, len(self.train_data))-i
      if batch_size < self.config.batch_size: break

      batch_idxs = data_idxs[i:i+self.config.batch_size]
      batch_data = [self.train_data[ix] for ix in batch_idxs]

      feed_dict = self.build_feed_dict(batch_data)
     
      loss_value, _ = sess.run([self.total_loss, self.train_op], feed_dict=feed_dict)
      loss_history.append(loss_value)
     
      step = step+1
      if verbose:
        sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
            self.train_data), np.mean(loss_history)))
        sys.stdout.flush()
          

    # statistics
    # Root Prediction
    train_preds = []
    for i in range(0, len(self.train_data), self.config.batch_size):
      batch_size = min(i+self.config.batch_size, len(self.train_data))-i
      if batch_size < self.config.batch_size: break
      batch_idxs = data_idxs[i:i+self.config.batch_size]
      batch_data = [self.train_data[ix] for ix in batch_idxs]
      feed_dict = self.build_feed_dict(batch_data)
      root_prediction = sess.run(self.pred, feed_dict=feed_dict)
      train_preds += root_prediction.tolist()

    val_preds = []
    val_losses = []
    data_idxs = list(range(len(self.dev_data)))
    for i in range(0, len(self.dev_data), self.config.batch_size):
      batch_size = min(i+self.config.batch_size, len(self.dev_data))-i
      if batch_size < self.config.batch_size: break
      batch_idxs = data_idxs[i:i+self.config.batch_size]
      batch_data = [self.dev_data[ix] for ix in batch_idxs]
      feed_dict = self.build_feed_dict(batch_data)

      root_prediction, loss = sess.run([self.pred, self.rloss], feed_dict=feed_dict)
      val_losses += loss.tolist()
      val_preds += root_prediction.tolist()

    train_labels = [t.root.label for t in self.train_data[0: (len(self.train_data)//self.config.batch_size)*self.config.batch_size]]
    val_labels = [t.root.label for t in self.dev_data[0: (len(self.dev_data)//self.config.batch_size)*self.config.batch_size]]
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
      #lr annealing
      epoch_loss = np.mean(loss_history)
      if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
        self.config.lr /= self.config.anneal_by
        print('annealed lr to %f' % self.config.lr)
      prev_epoch_loss = epoch_loss

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch

      # if model has not imprvoved for a while stop
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

    sess = tf.Session()

    start_time = time.time()
    stats = self.train(verbose=True, sess=sess)
    print('\nTraining time: {}'.format(time.time() - start_time))

    self.plot_loss_history(stats)
    
    start_time = time.time()

    val_preds = []
    val_losses = []

    data_idxs = list(range(len(self.dev_data)))
    for i in range(0, len(self.dev_data), self.config.batch_size):
      batch_size = min(i+self.config.batch_size, len(self.dev_data))-i
      if batch_size < self.config.batch_size: break
      batch_idxs = data_idxs[i:i+self.config.batch_size]
      batch_data = [self.dev_data[ix] for ix in batch_idxs]
      feed_dict = self.build_feed_dict(batch_data)
      root_prediction, loss = sess.run([self.pred, self.rloss], feed_dict=feed_dict)
      val_losses += loss.tolist()
      val_preds += root_prediction.tolist()

    val_labels = [t.root.label for t in self.dev_data[0: (len(self.dev_data)//self.config.batch_size)*self.config.batch_size]]
    val_acc = np.equal(val_preds, val_labels).mean()
    print(val_acc)
    
    print('-' * 20)
    print('Test')

    predictions = []

    data_idxs = list(range(len(self.test_data)))
    for i in range(0, len(self.test_data), self.config.batch_size):
      batch_size = min(i+self.config.batch_size, len(self.test_data))-i
      if batch_size < self.config.batch_size: break
      batch_idxs = data_idxs[i:i+self.config.batch_size]
      batch_data = [self.test_data[ix] for ix in batch_idxs]
      feed_dict = self.build_feed_dict(batch_data)
      root_prediction = sess.run(self.pred, feed_dict=feed_dict)
      predictions += root_prediction.tolist()

    labels = [t.root.label for t in self.test_data[0: (len(self.test_data)//self.config.batch_size)*self.config.batch_size]]
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