import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer, lr=0.01):

    # LOSS
        if isinstance(loss, str):
            if loss.lower() == "crossentropy":
                from ..losses.SoftmaxActivationCrossEntropyLoss import SoftmaxActivationCrossEntropyLoss
                self.loss = SoftmaxActivationCrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss: {loss}")
        else:
            self.loss = loss

        # OPTIMIZER
        if optimizer == 'sgd':
            from ..optim.sgd import StochasticGradientDescent
            self.optimizer = StochasticGradientDescent(learning_rate=lr)



    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def train(self, X, Y, epochs=1000, batch_size=None):
        samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(samples)
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            # Split into batches
            if batch_size is None:
                batch_size = samples  # full batch
            for start in range(0, samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                Y_batch = Y[start:end]

                # Forward
                output = self.forward(X_batch)
                softmaxloss = self.loss.forward(output)
                loss = self.loss.calculate(Y_batch, softmaxloss)

                # Backward
                self.loss.backward(Y_batch)
                dvalues = self.loss.dinputs
                self.backward(dvalues)

                # Update weights
                self.optimizer.pre_update_params()
                for layer in self.layers:
                    if hasattr(layer, "weights"):
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            # Display per epoch
            if epoch % 100 == 0:
                output_full = self.forward(X)
                acc = self.accuracy(Y, output_full)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}, lr: {self.optimizer.current_learning_rate:.6f}')

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, Y):
        output = self.forward(X)
        return self.loss.calculate(Y, output)

    def accuracy(self ,y_true, y_pred):
        # Convert predictions to class labels
        y_pred_labels = np.argmax(y_pred, axis=1)

        # If one-hot encoded → convert to labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        return np.mean(y_pred_labels == y_true)

