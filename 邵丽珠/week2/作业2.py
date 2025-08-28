import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.random.uniform(-2 * np.pi, 2 * np.pi, (1000, 1))  # 在[-2π, 2π]范围内生成数据
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上一些噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print(f"数据范围: x ∈ [{X_numpy.min():.2f}, {X_numpy.max():.2f}]")
print("---" * 10)


# 2. 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super(SinNet, self).__init__()
        # 三层网络结构: 输入层 → 隐藏层1 → 隐藏层2 → 输出层
        self.fc1 = torch.nn.Linear(1, hidden_size)  # 输入层到隐藏层1
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)  # 隐藏层1到隐藏层2
        self.fc3 = torch.nn.Linear(hidden_size, 1)  # 隐藏层2到输出层
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)  # 防止过拟合

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 3. 创建模型、损失函数和优化器
model = SinNet(hidden_size=128)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
print("---" * 10)

# 4. 训练模型
num_epochs = 2000
loss_history = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 评估模型
model.eval()  # 设置为评估模式
with torch.no_grad():
    # 生成测试数据用于绘制平滑曲线
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    x_test_tensor = torch.from_numpy(x_test).float()
    y_pred_test = model(x_test_tensor).numpy()

    # 计算训练数据上的预测
    y_pred_train = model(X).numpy()

# 6. 绘制结果
plt.figure(figsize=(10, 6))


# 绘制真实sin函数
x_true = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y_true = np.sin(x_true)
plt.plot(x_true, y_true, 'r-', linewidth=3, label='True sin(x)', alpha=0.8)

# 绘制训练数据点
plt.scatter(X_numpy, y_numpy, color='green', alpha=0.3, s=10, label='Training Data')

plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x) Function Fitting')
plt.legend()
plt.grid(True)
plt.show()
