#!/usr/bin/env python2.7

import load_mnist as lm
import numpy as np

class NeuralNet(object):
  def __init__(self, batch_size = 128, epsilon = .01, learning_rate = .01, num_iterations = 20):
    ''' initialize neural net, set params
        batch_size      -- number of training data to process at a time
        epsilon         -- scales multiply initial guess
        learning_rate   -- scales parameter update
        num_iterations  -- number of times to perform gradient descent

    '''
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.num_iterations = num_iterations

    self.input_layer = None
    self.output_layer = None
    self.predictions = None
    self.layers = []
    self.loss = None
    self.R = None
    self.class_indicator_matrix = None

  def add_layer(self, LayerType, **kwargs):
    '''user interface for call to  _add_new_layer'''
    self._add_new_layer(LayerType, **kwargs)

  def _add_new_layer(self, LayerType, **kwargs):
    ''' add new layer to neural net, set prev and next accordingly
        LayerType       -- either FullyConnected or Relu
        **kwargs        -- might include input_size & output_size
    '''
    try:
      prev = self.layers[-1]
    except:
      prev = None

    if LayerType == FullyConnectedLayer:
      new_layer = FullyConnectedLayer(self, kwargs['input_size'],
        kwargs['output_size'], self.epsilon, self.learning_rate, prev)
    if LayerType == ReluLayer:
      new_layer = ReluLayer(prev)

    if len(self.layers):
      self.layers[-1].next = new_layer
    self.layers.append(new_layer)

  def train(self, X, y):
    ''' train NeuralNet
        The output layer will send class scores.
        Calculate dP.
    '''
    self.set_class_indicator_matrix(y)
    for iteration in range(self.num_iterations):
      for batch in range(0, len(X), self.batch_size):
        X_batch = X[batch:batch + self.batch_size]
        y_batch = y[batch:batch + self.batch_size]
        self.layers[0].forward_input = X_batch
        self.forward()
        indices = range(batch, min(batch + self.batch_size, X.shape[0]))
        self.get_top_layer_backprop(indices, y_batch)
        self.backward()
        prediction = float((self.predict(X_batch).astype(int) == y_batch).sum())/len(X_batch)
      print "iteration {}: accuracy {:.4}".format(iteration, prediction)
 
  def forward(self):
    ''' call forward mthod for each layer in net '''
    for layer in self.layers:
      layer.forward()

  def backward(self):
    ''' call backward method on each layer in net '''
    for layer in reversed(self.layers):
      layer.backward()
    
  def set_class_indicator_matrix(self, y):
    ''' this is the matrix Y but for the entire data set.
        Store in self.class_indicator_matrix.'''
    R = np.arange(y.size)
    num_classes = np.unique(y).size
    self.class_indicator_matrix = np.zeros((y.size, num_classes))
    self.class_indicator_matrix[R, y] = 1
    
  def get_top_layer_backprop(self, indices, labels):
    ''' Get Y from self.class_indicator_matrix. Return dP '''
    top_layer = self.layers[-1]
    Y = self.class_indicator_matrix[indices]
    P = self.predictions
    E = np.exp(P)
    D = E / (E.sum(axis=1).reshape((len(indices), 1))) # sum over rows, shape
    gradient_P = (-Y + D) / len(indices)
    top_layer.backpropagated_input = gradient_P

  def predict(self, X):
    ''' return predictions for X '''
    self.layers[0].forward_input = X
    for layer in self.layers:
      layer.forward()
    return np.argmax(self.predictions, axis=1)

class FullyConnectedLayer(object):
  def __init__(self, master, input_size, output_size, epsilon, learning_rate, prev = None):
    ''' initialize layer and set parameters
        master          -- parent NeuralNet class
        input_size      -- number of input nodes
        output_size     -- number of output nodes
        learning_rate   -- scales parameter update
        prev            -- previous layer, if applicable
    '''
    self.master = master
    self.input_size = input_size
    self.output_size = output_size
    self.learning_rate = learning_rate
    self.prev = prev
    self.next = None

    self.backpropagated_input = None
    self.forward_input = None
    self.W = epsilon * np.random.randn(input_size, output_size)
    self.b = np.zeros(output_size)

  def forward(self):
    ''' perform forward pass and pass result to next layer
        if output layer, pass result to NeuralNet (master)
    '''
    prediction = np.dot(self.forward_input, self.W) + self.b
    if self.next is None:
      self.master.predictions = prediction
    else:
      self.next.forward_input = prediction
    
  def backward(self):
    ''' backpropagation, update, and pass data to prev layer
        if input layer, data is not passed
    '''
    gradient_W = np.dot(self.forward_input.T, self.backpropagated_input)
    gradient_b = np.sum(self.backpropagated_input, axis=0).reshape(self.b.shape)

    self.W -= self.learning_rate * gradient_W
    self.b -= self.learning_rate * gradient_b

    if self.prev is not None:
      gradient_X = np.dot(self.backpropagated_input, self.W.T)
      self.prev.backpropagated_input = gradient_X

class ReluLayer(object):
  ''' class for the Relu layer
      cannot be the first node of the Neural Net
  '''
  def __init__(self, prev = None):
    self.prev = prev
    self.next = None
    self.forward_input = None
    self.backpropagated_input = None

  def forward(self):
    ''' perform forward pass and pass result to next layer
        if output layer, pass result to NeuralNet
    '''
    prediction = np.multiply(self.forward_input, self.forward_input > 0)
    self.next.forward_input = prediction

  def backward(self):
    ''' backpropagation, update, and pass data to prev layer
        if input layer, data is not passed
    '''
    gradient_A = np.multiply(self.backpropagated_input, self.forward_input > 0)
    self.prev.backpropagated_input = gradient_A

# first test
Xcoords = np.random.randn(300).reshape(300, 1)
Ycoords = np.random.randn(300).reshape(300, 1)
Xtrain = np.zeros((300, 2))
Xtrain[:, 0] = Xcoords.flatten()
Xtrain[:, 1] = Ycoords.flatten()
ytrain = np.zeros(Xtrain.shape[0])

XtestCoords = np.random.randn(25).reshape(25, 1)
YtestCoords = np.random.randn(25).reshape(25, 1)
Xtest = np.zeros((25, 2))
Xtest[:, 0] = XtestCoords.flatten()
Xtest[:, 1] = YtestCoords.flatten()
ytest = np.zeros(Xtest.shape[0])

ytrain = (Xtrain[:,1] > 0).astype(int)
ytest = (Xtest[:,1] > 0).astype(int)

#######Softmax Classifier############   
#clf = NeuralNet(batch_size = 10, num_iterations = 100)
#clf.add_layer(FullyConnectedLayer, input_size = 2, output_size = 2)
#
#clf.train(Xtrain, ytrain)
#predictions = clf.predict(Xtest)
#results = (predictions == ytest)
##print results
#print "{} / {}".format(results.sum(), len(results))
#layers = clf.layers
#for layer in layers:
#  print layer.W
#  print layer.b

# one layer net
#clf2 = NeuralNet(batch_size=10, num_iterations = 50)
#clf2.add_layer(FullyConnectedLayer, input_size = 2, output_size = 10)
#clf2.add_layer(ReluLayer)
#clf2.add_layer(FullyConnectedLayer, input_size = 10, output_size = 2)
#clf2.train(Xtrain, ytrain)
#predictions = clf2.predict(Xtest)
#results = (predictions == ytest)
#print results.sum()
##for layer in clf2.layers:
#  print layer.prev, layer.next


# test mnist
mnist_train, mnist_labels, mnist_test, mnist_test_labels = lm.load_mnist()

def train_and_test(learning_rate = .0001, batch_size = 128, num_iterations = 50, epsilon = .01):
    clf_mnist = NeuralNet(learning_rate = learning_rate, batch_size =
    batch_size, num_iterations = num_iterations)
    clf_mnist.add_layer(FullyConnectedLayer, input_size = 784, output_size = 2000)
    clf_mnist.add_layer(ReluLayer)  
    clf_mnist.add_layer(FullyConnectedLayer, input_size = 2000, output_size = 10)
    clf_mnist.train(mnist_train, mnist_labels)
    predictions = clf_mnist.predict(mnist_test)
    results = (predictions == mnist_test_labels)
    print results.sum()

if __name__ == "__main__":
  train_and_test()
