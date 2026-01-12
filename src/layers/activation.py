import numpy as np

class ReluActivation:
    def __init__(self):
        pass

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def backword(self,dvalues):

        self.dinput = dvalues.copy()
        self.dinput[dvalues > 0]

         
    
class SigmoidActivation:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)  # x is sigmoid(x)
    
class TanhActivation:
    def __init__(self):
        pass

    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2  # x is tanh(x)

class SoftmaxActivation:
    def __init__(self):
        pass

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)  # Note: This is a simplification; the full Jacobian is more complex
    


    
class ActivationFactory:
    @staticmethod
    def get_activation(name):
        if name.lower() == 'relu':
            return ReluActivation()
        elif name.lower() == 'sigmoid':
            return SigmoidActivation()
        elif name.lower() == 'tanh':
            return TanhActivation()
        elif name.lower() == 'softmax':
            return SoftmaxActivation()
        else:
            raise ValueError(f"Unknown activation function: {name}")