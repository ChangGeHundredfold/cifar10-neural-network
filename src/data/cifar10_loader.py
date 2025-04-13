import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

def load_cifar10_batch(batch_filename):
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    return data, labels

def load_cifar10_data(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.append(labels)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    # Split train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)

    return (X_train, y_train), (X_val, y_val), (test_data, test_labels)

if __name__ == "__main__":
    data_dir = 'src/data/cifar-10-batches-py'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10_data(data_dir)
    print("CIFAR-10 data loaded successfully.")

