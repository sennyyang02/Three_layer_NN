# utils.py
import sys
import os

# 切换当前目录为 search 脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import numpy as np
import os
import pickle 
from urllib.request import urlretrieve
import tarfile

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_and_extract_cifar10(data_dir="../data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filepath = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(filepath):
        print("Downloading CIFAR-10 dataset...")
        urlretrieve(CIFAR_URL, filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("CIFAR-10 dataset extracted.")


def load_cifar10_batch(batch_path):
    with open(batch_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
        y = np.array(dict[b'labels'])
        X = X.transpose(0, 2, 3, 1).reshape(X.shape[0], -1) / 255.0
        return X, y


def load_cifar10(data_dir="../data/cifar-10-batches-py"):
    X_train, y_train = [], []
    for i in range(1, 6):
        X_batch, y_batch = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
        X_train.append(X_batch)
        y_train.append(y_batch)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    return X_train, y_train, X_test, y_test


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def train_val_split(X, y, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
