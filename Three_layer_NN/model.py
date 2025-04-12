# model.py
import numpy as np
import pickle

class ThreeLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', reg=0.0):
        self.reg = reg
        self.activation_type = activation
        self.params = {}
        self._init_weights(input_dim, hidden_dim, output_dim)

    def _init_weights(self, D, H, C):
        self.params['W1'] = np.random.randn(D, H) * 0.01
        self.params['b1'] = np.zeros((1, H))
        self.params['W2'] = np.random.randn(H, C) * 0.01
        self.params['b2'] = np.zeros((1, C))

    def _activation(self, z):
        if self.activation_type == 'relu':
            return np.maximum(0, z)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))

    def _activation_derivative(self, z):
        if self.activation_type == 'relu':
            return (z > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = X @ W1 + b1
        a1 = self._activation(z1)
        z2 = a1 @ W2 + b2
        out = self.softmax(z2)

        self.cache = (X, z1, a1, z2, out)
        return out

    def compute_loss(self, y_pred, y_true):
        N = y_true.shape[0]
        correct_logprobs = -np.log(y_pred[range(N), y_true] + 1e-8)
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss

    def backward(self, y_true):
        X, z1, a1, z2, y_pred = self.cache
        W1, W2 = self.params['W1'], self.params['W2']
        N = X.shape[0]

        dz2 = y_pred.copy()
        dz2[range(N), y_true] -= 1
        dz2 /= N

        dW2 = a1.T @ dz2 + self.reg * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * self._activation_derivative(z1)
        dW1 = X.T @ dz1 + self.reg * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return grads

    def update_params(self, grads, lr):
        for param in self.params:
            self.params[param] -= lr * grads[param]

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
