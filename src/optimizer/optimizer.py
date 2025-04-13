import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from src.model.neural_network import LinearLayer
class Optimizer:
    def __init__(self, model, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        Initialize the optimizer.
        Args:
            model: The neural network model.
            learning_rate: Learning rate for parameter updates.
            momentum: Momentum factor (default: 0.0).
            weight_decay: L2 regularization factor (default: 0.0).
        """
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(layer.weights) for layer in model.layers if isinstance(layer, LinearLayer)]


    def step(self):
        """
        Perform a single optimization step (update parameters).
        """
        k = 0
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LinearLayer):
                # Apply weight decay (L2 regularization)
                if self.weight_decay > 0:
                    layer.gradW += self.weight_decay * layer.weights

                # Update weights with momentum
                self.velocity[k] = self.momentum * self.velocity[k] - self.learning_rate * layer.gradW
                
                layer.weights += self.velocity[k]
                k += 1
                # Update biases
                layer.biases -= self.learning_rate * layer.gradB
                # print(np.linalg.norm(layer.gradB),np.linalg.norm(layer.gradW),'norm of grads')

    def zero_grad(self):
        """
        Clear the gradients of all layers.
        """
        for layer in self.model.layers:
            if isinstance(layer, LinearLayer):
                layer.gradW.fill(0)
                layer.gradB.fill(0)