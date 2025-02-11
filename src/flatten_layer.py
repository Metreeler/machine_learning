import numpy as np
from layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.output_shape = [None, None]
        
    def forward_propagation(self, input_vector):
        self.input_shape = input_vector.shape
        
        self.output_shape[0] = self.input_shape[0]
        
        out_dim = 1
        for i in range(1, len(self.input_shape), 1):
            out_dim *= self.input_shape[i]
        self.output_shape[1] = out_dim
        
        return np.reshape(input_vector, (self.output_shape))
    
    def backward_propagation(self, partial_derivative):
        return np.reshape(partial_derivative, self.input_shape)