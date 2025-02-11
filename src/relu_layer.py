import numpy as np
from layer import Layer

def sigmoid(array):
    return 1/(1 + np.exp(-array))

class ReluLayer(Layer):
    def __init__(self, threshold=None) -> None:
        super().__init__()
        self.threshold = threshold
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        if self.threshold is not None:
            np.where(input_vector > self.threshold, self.threshold, input_vector)
        return np.where(input_vector > 0, input_vector, 0)

    def backward_propagation(self, partial_derivative):
        if self.threshold is not None:
            return np.where((self.input_vector > 0) & (self.input_vector < self.threshold), 1, 0) * partial_derivative
        return np.where(self.input_vector > 0, 1, 0) * partial_derivative