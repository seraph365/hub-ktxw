# 作业：调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据
X_numpy = np.random.rand(100, 1) * 2 * np.pi
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 2π) 范围内均匀分布的随机浮点数。

y_numpy = 2 * np.sin(X_numpy) + 1 + np.random.randn(100, 1) * 0.1  #标准差最大为0.1
x = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层网络
class SinModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(SinModel, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(), # 增加模型的复杂度，非线性

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            # 第3层：从 hidden_size2 到 hidden_size3
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),

            # 输出层：从 hidden_size3 到 output_size
            nn.Linear(hidden_size3, output_size)
        )

    def forward(self, x):
        return self.network(x)

# --- 模型参数和实例化 ---
model = SinModel(input_size=1, hidden_size1=20, hidden_size2=30, hidden_size3=40, output_size=1)
print("模型结构:\n", model)
print("---" * 10)

# 3. 直接创建参数张量 a 和 b
# torch.randn() 生成随机值作为初始值。
# y = a * sin(x) + b
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)

# 4. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 优化器，基于 a b 梯度 自动更新

# 5. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * sin(x) + b
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每10个 epoch 打印一次损失
    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 6. 打印最终学到的参数
# print("\n训练完成！")
# a_learned = a.item()
# b_learned = b.item()
# print(f"拟合的斜率 a: {a_learned:.4f}")
# print(f"拟合的截距 b: {b_learned:.4f}")
# print("---" * 10)

# 7. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(x).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
sorted_idx = np.argsort(X_numpy, axis=0).flatten()
plt.plot(X_numpy[sorted_idx], y_predicted[sorted_idx], label='Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
