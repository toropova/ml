#!/usr/bin/python

import numpy as np
import pickle

#class NearestNeigbor(object):
#  def __init__(self):
#    pass
#
def load(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

Xtr, Ytr, Xte, Yte = load('data/cifar10/test_batch')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
print (Xtr_rows)
print ('***')
print (Xte_rows)

#  def train(self, X, y):
#    """ X is N x D where each row is an example. Y is 1-dimension of size N """
#    # the nearest neighbor classifier simply remembers all the training data
#    self.Xtr = X
#    self.ytr = y
#
#  def predict(self, X):
#    """ X is N x D where each row is an example we wish to predict label for """
#    num_test = X.shape[0]
#    # lets make sure that the output type matches the input type
#    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
#
#    # loop over all test rows
#    for i in xrange(num_test):
#      # find the nearest training image to the i'th test image
#      # using the L1 distance (sum of absolute value differences)
#      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
#      min_index = np.argmin(distances) # get the index with smallest distance
#      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
#
#    return Ypred

