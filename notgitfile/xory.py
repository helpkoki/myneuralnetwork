import numpy as np

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# helper functions
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(x): return x*(1-x)  # x is sigmoid(x)

# network shape: 2 -> 2 -> 1
np.random.seed(1)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.5
for epoch in range(10000): 
    # forward
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    # loss (mean squared)
    loss = np.mean((a2 - y) ** 2)

    # backprop
    d_a2 = 2*(a2 - y) / y.size
    d_z2 = d_a2 * sigmoid_deriv(a2)
    dW2 = a1.T.dot(d_z2)
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2.dot(W2.T)
    d_z1 = d_a1 * sigmoid_deriv(a1)
    dW1 = X.T.dot(d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 2000 == 0:
        print(type(z1))
        print(z1)
        print(f"epoch {epoch}, loss {loss:.5f}")

# test
print("predictions:")
print(np.round(a2, 3))
