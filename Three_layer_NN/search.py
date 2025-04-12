import sys
import os

# 切换当前目录为 search 脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import itertools
import numpy as np
from model import ThreeLayerNN
from utils import load_cifar10, train_val_split, accuracy
import matplotlib.pyplot as plt
import os

# 超参数搜索空间
learning_rates = [0.1, 0.01, 0.001]
regs = [0, 1e-4, 1e-3, 1e-2]
hidden_dims = [64, 128, 256]
activations = ['relu', 'sigmoid']

# CIFAR-10 数据加载与预处理
X_train_all, y_train_all, X_test, y_test = load_cifar10()
X_train, y_train, X_val, y_val = train_val_split(X_train_all, y_train_all)

results = []
best_val_acc = 0.0
best_model = None
best_config = None

# 迭代所有超参数组合
for lr, reg, hidden_dim, activation in itertools.product(learning_rates, regs, hidden_dims, activations):
    print(f"Training model with lr={lr}, reg={reg}, hidden_dim={hidden_dim}, activation={activation}")

    model = ThreeLayerNN(input_dim=3072, hidden_dim=hidden_dim, output_dim=10,
                         activation=activation, reg=reg)

    num_epochs = 20
    batch_size = 200
    train_losses = []

    for epoch in range(num_epochs):
        # 每个 epoch 使用 mini-batch SGD
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch)
            grads = model.backward(y_batch)
            model.update_params(grads, lr)

        # 验证集准确率
        val_preds = model.predict(X_val)
        val_acc = accuracy(val_preds, y_val)

        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.4f}")

    results.append((lr, reg, hidden_dim, activation, val_acc))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_config = (lr, reg, hidden_dim, activation)

# 输出最佳超参数组合
print("\nBest configuration:")
print(f"lr={best_config[0]}, reg={best_config[1]}, hidden_dim={best_config[2]}, activation={best_config[3]}")

# 保存最佳模型
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
best_model.save("checkpoints/best_model.pkl")

# 测试集表现
test_preds = best_model.predict(X_test)
test_acc = accuracy(test_preds, y_test)
print(f"Test Accuracy with best model: {test_acc:.4f}")

# 保存所有结果
with open("search_results.csv", "w") as f:
    f.write("lr,reg,hidden_dim,activation,val_acc\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f}\n")
