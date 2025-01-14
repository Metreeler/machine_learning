import numpy as np
from layer import Layer

def sigmoid(array):
    return 1/(1 + np.exp(-array))

class SigmoidLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        return sigmoid(self.input_vector)

    def backward_propagation(self, partial_derivative, _):
        return sigmoid(self.input_vector) * (1 - sigmoid(self.input_vector)) * partial_derivative