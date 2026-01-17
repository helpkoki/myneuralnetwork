import numpy as np 

inputs = np.array([1,2,3,4])


weights = np.array([
    [0.1,0.2,0.3,0.4],
    [0.5,0.6,0.7,0.8],
    [0.9,1.0,1.1,1.2]
])

biases =np.array([0.1,0.2,0.3])


learning_rate = 0.001 

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0 ,1 , 0)

for iteration in range(200):
    z =np.dot(weights ,input)+ biases 
    a =relu(z)
    y =np.sum(a)

    loss =y**2

    #backward pass
    #Gradient of the loss with respect to out put y
    dl_dy =2*y

    #Gradient of y with respect to a
    dy_da =np.ones_like(a)

    # Gradient of loss with respect to a
    dl_da= dl_dy*dy_da

    #Gradient of a with respect to z (ReLU , derivative)
    da_dz =relu_derivative(z)

    #calculation
    dl_dz  =dl_da*da_dz



    dl_dw = np.outer(dl_da,input)
    dl_db =dl_dz

      
     

