#!/usr/bin/env python2.7

import cPickle
import numpy as np, os

def load_file(file):
  with open(file, 'rb') as f:
    dict = cPickle.load(f)
  return np.asarray(dict['data']), np.asarray(dict['labels'])

def load_cifar10():
  train, train_labels = load_file('cifar10/data_batch_1')
  test, test_labels = load_file('cifar10/test_batch')

  return train, train_labels, test, test_labels

if __name__ == "__main__":
  load_cifar10()
