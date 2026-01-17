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
    
    def backward(self, y_true, y_pred=None):
            if y_true.ndim == 2:
                    y_true = np.argmax(y_true, axis=1)

            # Use y_pred if forward wasn't called
            if not hasattr(self, 'output'):
                if y_pred is None:
                    raise ValueError("Softmax backward needs y_pred if forward() was not called")
                self.output = y_pred

            samples = len(y_true)
            self.dinputs = self.output.copy()

            self.dinputs[range(samples), y_true] -= 1
            self.dinputs /= samples

            return self.dinputs


    def calculate(self , y_true , y_pred):
        self.loss=self.crossEntropyLoss.calculate(y_true , y_pred)
        return self.loss
    
    