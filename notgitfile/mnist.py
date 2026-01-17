from src import Model, Dense, ReluActivation, SGD , SoftmaxActivation
import numpy as np
import struct

def load_images(file):
    with open(file, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        return images / 255.0   # normalize

def load_labels(file):
    with open(file, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load data
X_train = load_images(
    "notgitfile/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte"
)

y_train = load_labels(
    "notgitfile/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)

X_test = load_images(
    "notgitfile/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
)

y_test = load_labels(
    "notgitfile/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
)


print(X_train.shape)  # (60000, 784)
print(y_train.shape)  # (60000,)
print(X_test[:1].shape)   # (10000, 784)

model = Model()
model.add(Dense(784, 128))
model.add(ReluActivation())
model.add(Dense(128, 128))
model.add(ReluActivation())
model.add(Dense(128, 10))

model.compile(
    loss='crossentropy',
    optimizer='sgd',
    lr=0.01
)
model.train(X_train, y_train, epochs=100001)

