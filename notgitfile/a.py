import numpy as np

# Example batch
X_batch = np.array([
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34]
])  # shape (B,4)

dL_dz_batch = np.array([
    [l11, l12, l13],
    [l21, l22, l23],
    [l31, l32, l33]
])  # shape (B,3)

# Gradient
dW = X_batch.T @ dL_dz_batch

print(dW.shape)  # (4,3)
print(dW)
