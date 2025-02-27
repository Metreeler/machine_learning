import numpy as np

from convolutionnal_layer import ConvolutionnalLayer


def loss_function(array, y):
    return (array - y) ** 2


def loss_function_derivative(array, y):
    return array - y


def accuracy(array, y):
    result = np.argmax(array, axis=1)
    expected_result = np.argmax(y, axis=1)
    acc = np.where(result == expected_result, 1, 0)
    
    return np.sum(acc) / acc.shape[0]


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, epochs, batch_proportion, x_values, y_values):
        
        batch_size = int(x_values.shape[0] * batch_proportion)
        batches = int(x_values.shape[0] / batch_size)
        
        for ep in range(epochs):
            print("Epoch :", ep, end=" ")
            idx = np.arange(x_values.shape[0])
            np.random.shuffle(idx)
            
            for i in range(batches):
                x = x_values[idx[(i * batch_size):((i + 1) * batch_size)]]
                y = y_values[idx[(i * batch_size):((i + 1) * batch_size)]]
                
                # Forward propagation
                x = self.forward(x)
                
                # Backward propagation
                self.backward(x, y)
            
            convolution_layer_count = 0
            for l in self.layers:
                if type(l) == ConvolutionnalLayer:
                    l.render_kernels(1, "data/kernels/" + str(convolution_layer_count) + "_" + str(ep) + ".png")
                    convolution_layer_count += 1
            
            
            print("loss :", np.sum(loss_function(x, y)))
    
    def test(self, x_values, y_values):
        out = self.forward(x_values)
        return accuracy(out, y_values)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x
    
    def backward(self, x, y):
        y = loss_function_derivative(x, y)
        for layer in reversed(self.layers):
            y = layer.backward_propagation(y)
        
        