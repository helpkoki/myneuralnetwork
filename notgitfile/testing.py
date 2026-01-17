import src.layers.dense as layer
import numpy as np
import nnit

# input_value = np.array([[0.5,0.2]
#                        ,[0.1,0.4]
#                        ,[0.3,0.8]])

# c = src.layers.dense.NeuralLayer(2, 2)
# print(c.weights)

# out = c.forward(input_value)
# print(out)


import src.losses.loss as loss

# softmax_outputs = np.array([[0.7, 0.1, 0.2],
#  [0.1, 0.5, 0.4],
#  [0.02, 0.9, 0.08]])
# class_targets = np.array([[1, 0, 0],
#  [0, 1, 0],
#  [0, 1, 0]])
# loss_function = loss.CrossEntropyLoss()
# loss = loss_function.calculate(softmax_outputs, class_targets)
# print(loss)


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print(activation2.output[:5])





