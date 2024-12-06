from layers import Dense
from loss import CrossEntropyLoss
from optimizer import Optimizer

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Optimizer(lr=0.01)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_pred, y_true):
        loss_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_fn.forward(y_pred, y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
            self.backward(y_pred, y)
            self.optimizer.step(self.layers[0])  # Update the first layer as an example
