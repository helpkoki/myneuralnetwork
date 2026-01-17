from src import Model, Dense, ReluActivation, SGD , SoftmaxActivation
from nnfs.datasets import spiral_data

X, y = spiral_data(100, 3)

model = Model()
model.add(Dense(2, 64))
model.add(ReluActivation())
model.add(Dense(64, 3))

model.compile(
    loss='crossentropy',
    optimizer='sgd',
    lr=0.01
)
model.train(X, y, epochs=1000)
