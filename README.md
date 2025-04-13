# Three-Layer Neural Network on CIFAR-10

本项目实现了一个使用 NumPy 构建的三层神经网络（无框架），用于对 CIFAR-10 图像数据集进行分类，并支持手动实现的超参数搜索与模型保存功能。

---

## 项目结构说明
ThreeLayerNN项目中主演包括三部分，分别为model.py,utils.py和search.py三个py文件，其中model为模型定义和前后向算法处理部分，utils为数据加载、预处理、辅助工具，search为训练主流程和超参数搜索。
除此以外在上传的文件中还有输出的文件，包括loss_curve.png和acc_curve.png，以及results.csv。

---

## 依赖安装

确保 Python 版本 ≥ 3.8,项目依赖：

```bash
pip install numpy matplotlib
```

## 数据集说明
使用 CIFAR-10 数据集，运行时自动下载并解压，无需手动准备数据：共 60000 张 32x32 彩色图像，分为 10 类；训练集：50000 张，测试集：10000 张

## 运行说明

运行以下命令，启动训练与超参数搜索过程：
python search.py
训练完成后会自动：1.输出最优超参数组合；2.保存最优模型权重到 checkpoints/best_model2.pkl；3.绘制 loss 和 accuracy 曲线图；4.导出搜索结果 CSV 到同一路径下 search_results2.csv。

    