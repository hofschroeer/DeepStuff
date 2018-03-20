#!/usr/bin/env python

import sys
import os
import numpy as np
np.random.seed(123)  # for reproducibility
import matplotlib.pyplot as plt

import tensorflow as tf

# set LOG_DIR based on this file's name ()
_, LOG_DIR = os.path.split(__file__)
LOG_DIR = 'TB_{}'.format(os.path.splitext(LOG_DIR)[0])

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def linearModel():
  """ Define a (poor) linear model """
  # other params
  batch_size = 4
  num_labels = 2

  graph = tf.Graph()
  with graph.as_default():
    # create placeholders
    with tf.name_scope('inputs'):
      _X = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='X')
      _y = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y')

    # create model params
    weights = tf.Variable(tf.random_uniform([num_labels, 1], -1, 1), name="weights")
    bias = tf.Variable(tf.zeros([1]), name="bias")
    with tf.name_scope('output'):
      output = tf.add(tf.matmul(_X, weights), bias)
    with tf.name_scope('loss'):
      loss = tf.losses.mean_squared_error(labels=_y, predictions=output)

    # # RMSProp Optimizer
    learning_rate = tf.constant(0.01, name='learning_rate')
    with tf.name_scope('optimizer'):
      optimizer = tf.train.RMSPropOptimizer(learning_rate, name='optimizer').minimize(loss)
  return graph, _X, _y, output, loss, learning_rate, optimizer

def layeredModel():
  """ A better model for XOR """
  # other params
  batch_size = 4
  num_labels = 2

  graph = tf.Graph()
  with graph.as_default():
    # create placeholders
    _X = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='X-input')
    _y = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

    # create model params
    with tf.name_scope('hidden'):
      hidden1 = tf.layers.dense(_X, 2, activation=tf.nn.relu, kernel_initializer=tf.random_uniform_initializer, name='hidden')
    with tf.name_scope('output'):
      output = tf.layers.dense(hidden1, 1, activation=tf.sigmoid, kernel_initializer=tf.random_uniform_initializer, name='output')
    with tf.name_scope('loss'):
      loss = tf.losses.mean_squared_error(labels=_y, predictions=output)

    # RMSProp Optimizer
    learning_rate = tf.constant(0.01, name='learning_rate')
    with tf.name_scope('optimizer'):
      optimizer = tf.train.RMSPropOptimizer(learning_rate, name='optimizer').minimize(loss)
  return graph, _X, _y, output, loss, learning_rate, optimizer

def run_model(model_fn, num_steps=1001):
  losses = []
  # create model and get required operations / tensors
  graph, _X, _y, output, loss, learning_rate, optimizer = model_fn()
  # write graph events to disk for using tensorboard
  tf.summary.FileWriter(logdir=LOG_DIR, graph=graph)
  # run session
  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for step in np.arange(num_steps):
        # feed the net with our inputs and desired outputs.
        feed_dict = {_X: X, _y: y }
        _, lr, l, out = sess.run([optimizer, learning_rate, loss, output], feed_dict=feed_dict)
        losses.append(l)
        if step % (num_steps // 10) == 0:
          print('Loss at step {}: {}'.format(step, l))
    print('Final output: \n{}'.format(out))
  return losses

# losses = run_model(linearModel)
# plt.figure()
# plt.plot(losses)
# plt.grid()
# plt.legend(['Losses'])
# plt.show()

losses = run_model(layeredModel, 1001)
plt.figure()
plt.plot(losses)
plt.grid()
plt.legend(['Losses'])
plt.show()