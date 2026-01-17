mytorch/
│
├── core/
│   ├── tensor.py               # Your custom Tensor class (optional)
│   ├── ops.py                  # Math operations (matmul, conv, etc.)
│   ├── autograd.py             # Backprop logic
│   └── utils.py                # Helpers
│
├── layers/
│   ├── dense.py                # Fully connected layer
│   ├── conv2d.py               # Convolution layer (optional)
│   └── activation.py           # ReLU, Sigmoid, Tanh, Softmax
│
├── losses/
│   └── losses.py               # MSE, CrossEntropy, etc.
│
├── optim/
│   ├── sgd.py                  # SGD optimizer
│   ├── adam.py                 # Adam optimizer
│   └── scheduler.py            # Learning rate scheduler
│
├── models/
│   └── model.py                # Sequential model class
│
├── data/
│   ├── dataloader.py           # Mini-batch loader
│   └── datasets.py             # MNIST loader or CSV loader
│
├── examples/
│   ├── train_mnist.py          # Use your library to train MNIST
│   └── xor_demo.py             # Your XOR “hello world”
│
├── tests/
│   └── test_layers.py          # Unit tests (optional)
│
└── README.md
