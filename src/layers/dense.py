import numpy as np


class NeuralLayer:
    def __init__(self, input_size, n_neurons):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        # Compute the output of the layer
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs