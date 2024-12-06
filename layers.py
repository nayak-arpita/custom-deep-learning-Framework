import numpy as np
from activation import ReLU, Softmax

class Dense:
    def __init__(self, input_size, output_size, activation=None):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.output = None
        self.dweights = None
        self.dbias = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def backward(self, doutput):
        # Backpropagation
        if self.activation:
            doutput = doutput * self.activation.derivative(self.output)
        
        self.dweights = np.dot(self.input.T, doutput)
        self.dbias = np.sum(doutput, axis=0, keepdims=True)
        return np.dot(doutput, self.weights.T)
