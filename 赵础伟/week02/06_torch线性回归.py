import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (sin 函数数据)
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)  # 从 0 到 2π，生成 100 个点
y_numpy = np.sin(X_numpy)+np.random.rand(100,1)  # 计算正弦值
X = torch.from_numpy(X_numpy).float()  # 转换为 PyTorch 张量
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, 64)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(64, 64)  # 隐藏层到隐藏层
        self.fc3 = torch.nn.Linear(64, 1)  # 隐藏层到输出层
        self.activation = torch.nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. 初始化模型、损失函数和优化器
model = SinNet()
criterion = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器

print("模型初始化完成。")
print("---" * 10)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
model.eval()  # 切换到评估模式
with torch.no_grad():
    y_predicted = model(X).numpy()  # 转换为 NumPy 数组

plt.figure(figsize=(10, 6))
plt.plot(X_numpy, y_numpy, label='True sin(x)', color='blue')  # 原始正弦函数
plt.plot(X_numpy, y_predicted, label='Predicted sin(x)', color='red')  # 拟合结果
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.legend()
plt.grid(True)
plt.title("Sin Function Fitting with Neural Network")
plt.show()