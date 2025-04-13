# CIFAR-10 Neural Network Classifier

This project implements a three-layer neural network classifier from scratch for image classification on the CIFAR-10 dataset. The neural network is built without using any deep learning frameworks like PyTorch or TensorFlow, relying solely on NumPy for numerical computations.

## Project Structure

```
cifar10-neural-network
├── src
│   ├── data
│   │   └── cifar10_loader.py       # Functions to load and preprocess the CIFAR-10 dataset
│   ├── model
│   │   └── neural_network.py        # NeuralNetwork class with forward and backward propagation
│   ├── training
│   │   └── train.py                 # Training loop with SGD, learning rate scheduling, and model saving
│   ├── testing
│   │   └── test.py                  # Evaluation of the trained model on the test dataset
│   └── optmizier
│   |    └── optimizer.py               # Implementation of SGD
|   ├── main_hypertuning.py              # Functions for hyperparameter tuning and performance recording
|   └── main.ipynb                       # Visualization of the training and testing of the model with best hyperparams 
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cifar10-neural-network
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the CIFAR-10 dataset and place it in the appropriate directory as specified in `cifar10_loader.py`.

## Usage

### Training the Model

To train the neural network, run the following command:
```
python src/training/train.py
```
This will initiate the training process, and the best model weights will be saved based on validation performance.

### Testing the Model

To evaluate the trained model on the test dataset, run:
```
python src/testing/test.py
```
This will load the saved model weights and compute the classification accuracy on the test set.

### Hyperparameter Tuning

To perform hyperparameter tuning, execute:
```
python src/main_hypertuning.py
```
This script will adjust various hyperparameters and record the model's performance for different configurations.


### Visualiztion

Run "src/main.ipynb" to see the training and testing results of the model with best hyper params.
## Implementation Details

- **Neural Network Architecture**: The neural network consists of an input layer, one hidden layer, and an output layer. The architecture allows customization of hidden layer sizes and activation functions.
- **Training Process**: The training loop implements stochastic gradient descent (SGD) with learning rate scheduling, cross-entropy loss calculation, and L2 regularization.
- **Evaluation Metrics**: The model's performance is evaluated using classification accuracy on the test dataset.

## Acknowledgments

This project is inspired by the principles of neural networks and deep learning. Special thanks to the contributors of the CIFAR-10 dataset.
