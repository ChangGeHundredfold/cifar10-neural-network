import numpy as np
import os
import pickle
import sys


sys.path.append(os.getcwd())
# print(sys.path)
# print(' ')
# print(os.path)

from src.model.neural_network import *
from src.data.cifar10_loader import load_cifar10_data
from src.optimizer.optimizer import Optimizer
def compute_loss(y_prob, y):
    m = y.shape[0]
    log_probs = -np.log(1e-15+y_prob[range(m), y])
    data_loss = np.sum(log_probs) / m
    # reg_loss = 0.5 * self.reg_lambda * sum(np.sum(w ** 2) for w in self.weights)/m
    return data_loss

def d_loss(y_prob, y_onehot):
    ## gradient of cross loss with respect to the input of the softmax layer
    return y_prob - y_onehot


def train_model(X_train, y_train, X_val, y_val,hidden_layer_sizes, learning_rate, momentum,weight_decay,num_epochs, batch_size, l2_reg, save_path, activation, init_method):
    # Initialize the neural network
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=hidden_layer_sizes,  output_size=10, activation = activation, reg_lambda=l2_reg,init_method=init_method)
    optimizer  = Optimizer(model, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
    best_val_accuracy = 0
    best_weights = None
    train_loss = []
    val_acc = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Shuffle the training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            output = model.forward(X_batch)
            if np.isnan(output).any():
                print("NaN detected in output data")
                exit()

            # Compute loss and gradients
            loss = compute_loss(output, y_batch)
            if np.isnan(loss).any():
                print("NaN detected in loss data")
                exit()

            epoch_loss += loss


            y_onehot = np.zeros((X_batch.shape[0], 10))
            y_onehot[np.arange(X_batch.shape[0]), y_batch] = 1
            dz = d_loss(output, y_onehot)
            model.backward(dz)

            # Update weights using SGD
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()
        epoch_loss /= (X_train.shape[0] // batch_size)
        train_loss.append(epoch_loss)
        # Validate the model
        val_predictions = model.forward(X_val)
        val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == y_val)
        val_acc.append(val_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = {}
            for i, layer in enumerate(model.layers):
                if isinstance(layer, LinearLayer):
                    best_weights[f'layer_{i}'] = layer.weights
                    best_weights[f'bias_{i}'] = layer.biases

        

    # Save the best model weights
    if save_path and best_weights is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(best_weights, f)
        print(f'Model weights saved to {save_path}')
        print(f'Best validation accuracy: {best_val_accuracy:.4f}')

    return train_loss,val_acc,best_val_accuracy



