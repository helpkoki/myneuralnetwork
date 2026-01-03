import numpy as np
import src.layers.dense

layoutputs = [[4.8 ,1.21,2.385]]
layoutputs =np.array(layoutputs)
def sigmoid(x): return 1/(1+np.exp(-x))

x =sigmoid(layoutputs)
print(np.random.randn(3,2)*0.1)




