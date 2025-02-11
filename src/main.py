import numpy as np

from flat_layer import FlatLayer
from softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork
from sigmoid_layer import SigmoidLayer
from relu_layer import ReluLayer
from convolutionnal_layer import ConvolutionnalLayer
from flatten_layer import FlattenLayer


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
    
    x_train = np.reshape(x_train, (60000, 1, 28, 28))
    x_test = np.reshape(x_test, (10000, 1, 28, 28))
    
    print(x_train.shape)
    print(y_train.shape)
    
    layers = [ConvolutionnalLayer(1, 3, 5, 2, 0.03), # 3 x 12 x 12
              ConvolutionnalLayer(3, 9, 3, 2, 0.03), # 9 x 5 x 5
              FlattenLayer(),
              FlatLayer(225, 50, 0.0001),
              SigmoidLayer(),
              FlatLayer(50, 10, 0.0001),
              SoftmaxLayer()]
    
    nn = NeuralNetwork(layers)
    
    nn.train(20, 0.01, x_train, y_train)
    
    print(nn.test(x_test, y_test))