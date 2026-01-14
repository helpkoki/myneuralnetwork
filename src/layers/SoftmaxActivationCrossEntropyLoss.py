import numpy as np
import src.losses.loss as loss

class SoftmaxActivationCrossEntropyLoss:
    def __init__(self):
        self.crossEntropyLoss =loss.CrossEntropyLoss()

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, y_true):
        samples = len(y_true)
        self.dinput = self.output.copy()
        self.dinput[range(samples), y_true] -= 1
        self.dinput /= samples
        return self.dinput
    
    def calculate(self , y_true , y_pred):
        self.loss=self.crossEntropyLoss.calculate(y_true , y_pred)
        return self.loss
    
    