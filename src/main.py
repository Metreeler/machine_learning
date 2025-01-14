import numpy as np

from flat_layer import FlatLayer
from softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork
from sigmoid_layer import SigmoidLayer
from relu_layer import ReluLayer


def load_dataset(path):
    data = np.loadtxt(path, delimiter=",", dtype=int, skiprows=1)
    x = data[:, 1:]
    y = data[:, 0]
    
    max_value = np.unique(y).size
    y = np.eye(max_value)[y]
    
    return x.astype(float), y


def loss_function(array, y):
    return (array - y) ** 2


def loss_function_derivative(array, y):
    return array - y


if __name__ == "__main__":
    print("Loading Data")
    x_train, y_train = load_dataset("data/mnist_train.csv")
    x_test, y_test = load_dataset("data/mnist_test.csv")
    x_train /= 255
    x_test /= 255
    
    layers = [FlatLayer(x_train.shape[1], 800),
              ReluLayer(),
              FlatLayer(800, 10),
              SoftmaxLayer()]
    
    nn = NeuralNetwork(layers)
    
    nn.train(100, 0.00001, 0.1, x_train, y_train)
    
    print(nn.test(x_train, y_train))