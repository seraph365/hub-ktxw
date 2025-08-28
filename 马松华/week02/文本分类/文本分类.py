#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/8/21 20:50
 Description:
1.调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
"""
from typing import Any

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import ndarray, dtype

X_numpy: ndarray[Any, dtype[Any]] = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 0 到 2π 的 1000 个点
y_numpy = np.sin(X_numpy)
# 收集所有X, 转换成tensor
X_tensor = torch.from_numpy(X_numpy).float()
y_tensor = torch.from_numpy(y_numpy).float()


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SimpleClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim1, output_dim):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


input_dim = 1
hidden_dim = 64
hidden_dim1 = 64
output_dim = 1

# model = SimpleClassifier(input_dim, hidden_dim, output_dim)
model = SimpleClassifier2(input_dim, hidden_dim, hidden_dim1, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predicted = model(X_tensor).numpy()

# 3. 绘图
# plt.figure(figsize=(10, 6))
plt.plot(X_numpy, y_numpy, label='True sin(x)')
plt.plot(X_numpy, predicted, label='Predicted sin(x)', linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title("sin(x)")
plt.show()

