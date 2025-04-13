# %%
import numpy as np
import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
from model.neural_network import *
from training.train import train_model
from testing.test import evaluate_model
from matplotlib import pyplot as plt
from data.cifar10_loader import load_cifar10_data
from model.neural_network import NeuralNetwork

sys.path.append('\\'.join([os.getcwd(),'src']))
print(sys.path)

# %%
num_epochs = 20
batch_size = 64

save_path = 'best_model_weights.pkl'
data_dir = 'src/data/cifar-10-batches-py'
weight_decay = 0
momentum = 0.0

# Load CIFAR-10 data
(X_train, y_train), (X_val, y_val),  (X_test,y_test)= load_cifar10_data(data_dir=data_dir)


# %%
def hyperparameter_tuning(X_train, y_train, X_val, y_val, learning_rates, hidden_layer_sizes, l2_regs,activations,init_methods):
    best_accuracy = -np.inf
    best_params = None
    for lr in learning_rates:
        for hls in hidden_layer_sizes:
            for l2_reg in l2_regs:
                for act in activations:
                    for init in init_methods:
                        print(f"Training with lr={lr}, hls={hls}, l2_reg={l2_reg}, act={act}, init={init}")
                        # Initialize the model with the current hyperparameters
                        # Train the model with the current hyperparameters

                        _,_,val_acc = train_model(X_train, y_train, X_val, y_val, hls, lr,momentum,weight_decay, num_epochs, batch_size, l2_reg, save_path, activation=act, init_method=init)




                        # Update best parameters if needed
                        if val_acc > best_accuracy:
                            best_accuracy = val_acc
                            best_params = {
                                'lr': lr,
                                'hls': hls,
                                'l2_reg': l2_reg,
                                'act': act,
                                'init': init
                            }
    return best_params, best_accuracy

    

# %%
# 定义一组要测试的超参数
learning_rates = [1e-7,1e-6,1e-5 ]
hidden_layer_sizes = [[i,j] for i in [512, 256] for j in [512, 256]]
# 这里的hidden_layer_sizes是一个二维列表，表示每个隐藏层的大小组合
l2_regs = [1e-4,1e-6]
activations = [ReLU,Tanh]
init_methods = ['random', 'xavier']
# 这里的activations是一个列表，表示要测试的激活函数组合

# 超参数搜索和训练
best_params, best_val_acc = hyperparameter_tuning(X_train, y_train, X_val, y_val, learning_rates, hidden_layer_sizes, l2_regs,activations,init_methods)


# %%
## 根据最佳超参数训练模型
print(f"Best hyperparameters: {best_params}, Best validation accuracy: {best_val_acc}")


## 保存最佳超参数
with open('best_hyperparameters.txt', 'w') as f:
    f.write(f"Best hyperparameters: {best_params}\n")

