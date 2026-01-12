import src.layers.dense as dense
import src.layers.activation as activation 
import   src.losses.loss as loss
from nnfs.datasets import spiral_data
import numpy as np

X, y = spiral_data(samples=100, classes=3)
w1 =dense.NeuralLayer(2 ,3)
w2 =dense.NeuralLayer(3 ,3)

loss_west=9999999

activate1 =activation.ReluActivation()
activate2 =activation.SoftmaxActivation()

output1=activate1.relu(w1.forward(X))
output2 = activate2.softmax(w2.forward(output1))
y_pred_clipped = np.clip(y, 1e-7, 1 - 1e-7)
y_true_size = len(y)
n_loss =loss.CrossEntropyLoss()
# print( len(output2.shape )==2)
# print(output2.shape)
# print(len(y_pred_clipped.shape) ==2)
# print(y.shape)


loss=n_loss.calculate(y , output2)
predictions = np.argmax(output2, axis=1)
accuracy = np.mean( predictions == y)


# for i in range(10000000000):
#     if loss < loss_west:
#         print('New set of weights found, iteration:', i ,'loss:', loss, 'acc:', accuracy)
#         best_weight_layer_one =w1.weights.copy()
#         best_weight_layer_two =w2.weights.copy()
#         best_base_layer_one =w1.biases.copy()
#         best_base_layer_two =w2.biases.copy()
#         loss_west=loss
#     else:
#         w1.weights =best_weight_layer_one
#         w2.weights =best_weight_layer_two
#         w1.biases =best_base_layer_one
#         w2.biases =best_base_layer_two

#     w1.weights +=np.random.randn(2, 3) * 0.01
#     w2.weights +=np.random.randn(3, 3) * 0.01
#     w1.biases +=np.zeros((1, 3))
#     w2.biases +=np.zeros((1, 3))
  
#     activate1 =activation.ReluActivation()
#     activate2 =activation.SoftmaxActivation()
   
#     output1=activate1.relu(w1.forward(X))
#     output2 = activate2.softmax(w2.forward(output1))
#     y_pred_clipped = np.clip(y, 1e-7, 1 - 1e-7)
#     y_true_size = len(y)
#     loss=n_loss.calculate(y , output2)
#     predictions = np.argmax(output2, axis=1)
#     accuracy = np.mean( predictions == y)
    
    
       



# y_true =np.eye()
print(output1[:5])





