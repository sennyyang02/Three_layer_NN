import sys
import os

# 切换当前目录为 search 脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import matplotlib.pyplot as plt
import numpy as np
import pickle

# 加载模型
with open("checkpoints/best_model2.pkl", "rb") as f:
    params = pickle.load(f)

W1 = params["W1"]  # (3072, 128)
W1_imgs = W1.T.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (128, 32, 32, 3)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow((W1_imgs[i] - W1_imgs[i].min()) / (W1_imgs[i].ptp()))  # 归一化
    plt.axis('off')
plt.suptitle("First 10 Hidden Neuron Weights Visualized as Images")
plt.tight_layout()
plt.savefig("weights_vis.png")
plt.show()