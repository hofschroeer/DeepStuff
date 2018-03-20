#!/usr/bin/env python

import matplotlib.pyplot as plt

import argparse

import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop

def run(params):
  # input / target values
  X = np.array([[0,0],[0,1],[1,0],[1,1]])
  y = np.array([[0],[1],[1],[0]])

  # model setup
  model = Sequential()
  model.add(Dense(4, input_dim=2))
  model.add(Activation('tanh'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  # configurable params
  print('Using learning rate: {}'.format(params.learning_rate))
  opt = None
  # ONLY SGD and RMSProp usable for now
  if params.optimizer == 'rms':
    opt = RMSprop(lr=params.learning_rate)
  else:
    opt = SGD(lr=params.learning_rate)
  print('Using optimizer: {}'.format(type(opt)))
  print('Using loss function: {}'.format(params.loss_fn))

  # create model
  model.compile(loss=params.loss_fn, optimizer=opt, metrics=['accuracy'])
  print(model.summary())

  if not params.skip_training:
    # train model
    train_history = model.fit(X, y, batch_size=4, epochs=params.num_epochs)
    # print final probs
    print(model.predict_proba(X))
    # print final weights
    # print(model.get_weights())
    if params.plot:
      # plot loss and accuracy
      plt.figure()
      plt.plot(train_history.history['loss'])
      plt.plot(train_history.history['acc'])
      plt.grid()
      plt.legend(['Loss', 'Accuracy'])
      plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-n',
    '--num_epochs',
    type=int,
    default=100,
    help='Number of training epochs.'
  )
  parser.add_argument(
    '-lr',
    '--learning_rate',
    type=float,
    default=0.01,
    help='Initial learning rate.'
  )
  parser.add_argument(
    '-o',
    '--optimizer',
    type=str,
    default='rms',
    help='Used optimizer.'
  )
  parser.add_argument(
    '-l',
    '--loss_fn',
    type=str,
    default='mse',
    help='Used loss function.'
  )
  parser.add_argument(
    '--skip-training',
    type=bool,
    default=False,
    help='Skip trainning (only show model).'
  )
  parser.add_argument(
    '--plot',
    type=bool,
    default=False,
    help='Plot loss and accuracy after training.'
  )
  params, _ = parser.parse_known_args()
  run(params)