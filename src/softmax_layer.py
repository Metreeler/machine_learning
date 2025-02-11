import numpy as np
from layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_vector = np.array([])
    
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        e_x = np.exp(input_vector)
        out = e_x / np.sum(e_x, axis=1)[:, np.newaxis]
        return out

    def backward_propagation(self, partial_derivative):
        return partial_derivative
