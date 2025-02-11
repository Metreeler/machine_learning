import numpy as np


class Layer:
    def __init__(self):
        self.input_vector = np.array([])
        
    def forward_propagation(self, input_vector):
        self.input_vector = input_vector
        return np.array([])
    
    def backward_propagation(self, partial_derivative):
        return np.array([])