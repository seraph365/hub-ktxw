import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 生成sin函数模拟数据
X_numpy = np.random.rand(100, 1) * 10  # 生成0到10之间的随机数
# 生成sin函数值并添加一些噪声
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# 初始化模型
input_dim = 1  # 输入特征维度
hidden_dim1 = 128  # 第一个隐藏层节点数
hidden_dim2 = 32  # 第二个隐藏层节点数
output_dim = 1  # 输出维度

model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
print("多层神经网络模型初始化完成。")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印训练完成信息
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    # 生成用于绘图的连续x值
    x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    x_plot_tensor = torch.from_numpy(x_plot).float()
    y_predicted = model(x_plot_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='原始数据', color='blue', alpha=0.6)
plt.plot(x_plot, np.sin(x_plot), label='真实sin函数', color='green', linewidth=2)
plt.plot(x_plot, y_predicted, label='神经网络拟合', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('多层神经网络拟合sin函数')
plt.show()

# 7. 保存模型（可选）
# torch.save(model.state_dict(), 'sin_model.pth')
print("模型拟合和可视化完成。")