class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.bias -= self.lr * layer.dbias
