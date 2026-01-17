import src.layers.dense as nr
import src.losses.loss as loss
import src.layers.activation as activation 
import src.losses.SoftmaxActivationCrossEntropyLoss as sft 
import src.optim.sgd as sgd
import numpy as np
from nnfs.datasets import spiral_data

X, Y = spiral_data(samples=100, classes=3)

w1 =nr.NeuralLayer(2 ,64)
w2 =nr.NeuralLayer(64,3)

activate1 =activation.ReluActivation()
activate2=sft.SoftmaxActivationCrossEntropyLoss()

otm =sgd.StochasticGradientDescent ()

output1=activate1.forward(w1.forward(X))
output2=activate2.forward(w2.forward(output1))

#for traacking 
loss=activate2.calculate(Y , output2)
predictions = np.argmax(output2, axis=1)
accuracy = np.mean( predictions == Y)

  

for i in range(1001):
    
    dvalues =activate2.backward(Y)
    dvalues=w2.backward(dvalues)
    dvalues =activate1.backward(dvalues)
    dvalues=w1.backward(dvalues)
   

    if i % 100 == 0:
        print(f'epoch: {i}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {otm.current_learning_rate}')

    
    otm.pre_update_params()
    otm.update_params(w2)
    otm.update_params(w1)
    otm.post_update_params()
    output1=activate1.forward(w1.forward(X))
    output2=activate2.forward(w2.forward(output1))

    loss=activate2.calculate(Y , output2)
    predictions = np.argmax(output2, axis=1)
    accuracy = np.mean( predictions == Y)



