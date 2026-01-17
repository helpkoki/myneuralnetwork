import numpy as np
import src.layers.dense as layer

layoutputs = [[4.8 ,1.21,2.385]]
layoutputs =np.array(layoutputs)
def sigmoid(x): return 1/(1+np.exp(-x))

# x =sigmoid(layoutputs)
# print(np.random.randn(3,2)*0.1)
# x = layer.NeuralLayer(3 , 3) 
# print("w"*3)
# print(x.weights)
# print("w"*3)

# print(x.forward(layoutputs))



one = np.array([1, 0])

two = np.array([
    [3.2, 0.3, 1.9],
    [3.2, 0.3, 1.9]
])

plus = two[range(len(two)), one]
idx = np.argmax(two, axis=1)
print(idx)
