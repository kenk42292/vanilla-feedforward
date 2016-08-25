import cPickle
import gzip

import numpy as np

"""
Returns the MNIST data-set.
TRAINING_DATA: a list of tuples of the training set: Each tuple is a couple,\
                consisting of the 784 pixel values, and the 10-long output vector
VALIDATION_DATA: a list of couples of val set: Each couple consists of 784-pixel value to single digit output.
TEST_DATA: a list of couples of test set: same format as validation_data.
"""

def int2Onehot(x, size):
    result = np.zeros((size, 1))
    result[x] = 1.0
    return result

def load_mnist(data_path='./datasets/mnist.pkl.gz'):
    f = gzip.open(data_path, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    training_data = np.array(
        [(training_data[0][i], int2Onehot(training_data[1][i], 10)) for i in range(len(training_data[0]))])
    validation_data = np.array([(validation_data[0][i], validation_data[1][i]) for i in range(len(validation_data[0]))])
    test_data = np.array([(test_data[0][i], test_data[1][i]) for i in range(len(test_data[0]))])
    return training_data, validation_data, test_data
