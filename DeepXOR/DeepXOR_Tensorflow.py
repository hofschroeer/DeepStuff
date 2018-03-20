#!/usr/bin/env python

import sys
import numpy as np
np.random.seed(123)  # for reproducibility
import matplotlib.pyplot as plt

import tensorflow as tf

batch_size = 4
num_labels = 2

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

graph = tf.Graph()
graph.as_default()
_X = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='x-input')
_y = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

def simpleModel(graph):
  graph.as_default()

  weights = tf.Variable(tf.random_uniform([num_labels, 1], -1, 1), name="weights")
  bias = tf.Variable(tf.zeros([num_labels]), name="bias")
  output = tf.reduce_sum(tf.nn.softmax(tf.matmul(_X, weights) + bias), axis=1, keepdims=True, name='output')
  loss = tf.losses.mean_squared_error(labels=_y, predictions=output)

  global_step = tf.Variable(0, name='global_step')
  # RMSProp Optimizer
  learning_rate = tf.constant(0.01)
  optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# run session
simpleModel(graph)
tf.initialize_all_variables().run()
sess = tf.InteractiveSession(graph=graph)
for step in range(1000):
    feed_dict={_X: X, _y: y } # feed the net with our inputs and desired outputs.
    _, loss,step=sess.run([optimizerm, loss, global_step],feed_dict)