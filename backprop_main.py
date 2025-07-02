import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

#some experiments of mine: (to be deleted)
net_tmp = Network([1])          # tiny throw-away network
x = np.array([[-2, 0, 3],
              [ 5,-1, 4]])
print(net_tmp.relu(x))          # should print [[0 0 3] [5 0 4]]

v = np.array([[-2.0, 0.0, 3.0],
              [ 5.0,-1.0, 4.0]])
print(net_tmp.relu_derivative(v))
# Expected:
# [[0. 0. 1.]
#  [1. 0. 1.]]


# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# Training configuration
epochs = 30
batch_size = 100
learning_rate = 0.1

# Network configuration
layer_dims = [784, 40, 10]
net = Network(layer_dims)
net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)
