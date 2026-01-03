import src.layers.dense
import numpy as np

input_value = np.array([[0.5,0.2]
                       ,[0.1,0.4]
                       ,[0.3,0.8]])

c = src.layers.dense.NeuralLayer(2, 2)
print(c.weights)

out = c.forward(input_value)
print(out)









