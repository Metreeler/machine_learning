import numpy as np
from layer import Layer

class FlatLayer(Layer):
    def __init__(self, input_layer_size, layer_size, learning_rate):
        super().__init__()
        self.weights = np.random.rand(input_layer_size, layer_size) * 2 - 1
        self.biases = np.random.rand(1, layer_size) * 2 - 1
        self.input_vector = np.array([])
        self.learning_rate = learning_rate
        
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        out = np.matmul(input_vector, self.weights) + self.biases
        return out

    def backward_propagation(self, partial_derivative):
        old_weights = self.weights
        self.weights -= np.matmul(self.input_vector.T, partial_derivative) * self.learning_rate
        self.biases -= np.mean(partial_derivative, axis=0) * self.learning_rate
        return np.dot(partial_derivative, old_weights.T)
    
    