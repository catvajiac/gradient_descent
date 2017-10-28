import numpy as np
import os

def load_mnist():
  if os.path.isfile('mnist/mnist_train.npy'):
    train = np.load('mnist/mnist_train.npy')
    train_labels = np.load('mnist/mnist_train_labels.npy')
    test = np.load('mnist/mnist_test.npy')
    test_labels = np.load('mnist/mnist_test_labels.npy')
  else:
    train = np.loadtxt('mnist/full_mnist_train.csv', delimiter = ',',skiprows = 1)
    train_labels = np.loadtxt('mnist/full_mnist_train_labels.csv')[1:].astype(int)
    test = np.loadtxt('mnist/full_mnist_test.csv', delimiter = ',', skiprows = 1)
    test_labels = np.loadtxt('mnist/full_mnist_test_labels.csv')[1:].astype(int) 
  
    np.save("mnist/mnist_train", train)
    np.save("mnist/mnist_train_labels", train_labels)
    np.save("mnist/mnist_test", test)
    np.save("mnist/mnist_test_labels", test_labels)

  return train, train_labels, test, test_labels

if __name__ == "__main__":
  load_mnist()
