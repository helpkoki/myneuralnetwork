# Neural Network From Scratch This repository implements a simple neural network framework from scratch using **NumPy**. Below is the organized project structure:

neural-network-from-scratch/
│
├── data/
│   ├── raw/                  # Original datasets (MNIST, custom images)
│   └── processed/            # Normalized / flattened versions
│
├── src/
│   ├── __init__.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── dense.py          # Fully connected layer
│   │   └── dropout.py        # Optional dropout layer
│   │
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   ├── softmax.py
│   │   └── activation_base.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── mse.py
│   │   └── cross_entropy.py
│   │
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── sgd.py
│   │   └── adam.py
│   │
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── visualize.py      # Plot loss, accuracy
│   │   └── metrics.py        # Accuracy, precision, etc.
│   │
│   ├── model.py              # Neural network class (forward + backward)
│   └── train.py              # Training loop
│
├── api/
│   ├── app.py                # Flask API (optional)
│   └── model_loader.py       # Load trained weights
│
├── models/
│   ├── checkpoints/          # Save model during training
│   └── final/                # Final trained weights
│
├── notebooks/
│   ├── experiments.ipynb     # For testing ideas
│   └── data_exploration.ipynb
│
├── tests/
│   ├── test_layers.py
│   ├── test_activations.py
│   └── test_model.py
│
├── results/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── confusion_matrix.png
│
├── requirements.txt          # Libraries: numpy, matplotlib, flask...
├── README.md                 # Project documentation
└── run.py                    # Main entry point (train or test)
