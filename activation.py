import numpy as np

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Softmax:
    def __call__(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def derivative(self, x):
        # Simplified version of the softmax derivative
        return np.outer(x, 1 - x)  # For educational purposes, not fully optimized
