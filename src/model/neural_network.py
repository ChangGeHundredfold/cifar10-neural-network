import numpy as np
import pickle
class LinearLayer:
    def __init__(self, input_size, output_size,init_method='random'):
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights and biases
        if init_method == 'random':
            self.weights = np.random.randn(input_size, output_size) * 0.01
        elif init_method == 'xavier':
            # Xavier 初始化
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        else:
            raise ValueError(f"Unsupported initialization method: {self.init_method}")
            
        self.biases = np.zeros((1, output_size))
        self.gradW = np.zeros((input_size, output_size))
        self.gradB = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        z = np.dot(X, self.weights) + self.biases
        self.output = z
        return self.output
    def backward(self, dz):
        self.gradW = np.dot(self.input.T, dz)
        self.gradB = np.sum(dz, axis=0, keepdims=True)
        dX = np.dot(dz, self.weights.T)
        return dX

class ReLU:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.maximum(0, X)
        return self.output

    def backward(self, dz):
        if np.isnan(dz).any():
            print("NaN detected in dz data")
        if np.isnan(self.input).any():
            print("NaN detected in input data")
        dX = dz * (self.input > 0)
        return dX

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, dz):
        dX = dz * (self.output * (1 - self.output))
        return dX


class Tanh:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.tanh(X)
        return self.output

    def backward(self, dz):
        dX = dz * (1 - np.square(self.output))
        return dX
def softmax(X):
    if np.isnan(X).any():
        print("NaN detected in loss data")

    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation=ReLU, reg_lambda=0.0, init_method='random'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = [LinearLayer(input_size, hidden_sizes[0],init_method=init_method), activation()]
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(LinearLayer(hidden_sizes[i], hidden_sizes[i + 1],init_method=init_method))
            self.layers.append(activation())
        self.layers.append(LinearLayer(hidden_sizes[-1], output_size,init_method=init_method))

        self.reg_lambda = reg_lambda
        # for ele in self.layers:
        #     if isinstance(ele, LinearLayer):
        #         print(ele.weights.shape)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return softmax(X)

    def backward(self, dz):
        for layer in reversed(self.layers):
            dz = layer.backward(dz)
        return dz

    def get_weights_grads(self):
        weights_grads = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                weights_grads.append((layer.weights, layer.gradW))
                weights_grads.append((layer.biases, layer.gradB))
        return weights_grads


    # def zero_grads(self):
    #     for layer in self.layers:
    #         if isinstance(layer, LinearLayer):
    #             layer.gradW.fill(0)
    #             layer.gradB.fill(0)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def load_weights(self, save_path):
        with open(save_path, 'rb') as f:
            best_weights = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                layer.weights = best_weights[f'layer_{i}']
                layer.biases = best_weights[f'bias_{i}']
        print(f'Model weights loaded from {save_path}')
 






