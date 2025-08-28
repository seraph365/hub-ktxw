import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成sin函数模拟数据
X_numpy = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)  # 生成从-π到π的1000个点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # 生成带噪声的sin函数值

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self, hidden_size=64):
        super(SinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),  # 输入层到隐藏层
            nn.Tanh(),                  # 激活函数
            nn.Linear(hidden_size, hidden_size),  # 隐藏层到隐藏层
            nn.Tanh(),                  # 激活函数
            nn.Linear(hidden_size, 1)   # 隐藏层到输出层
        )
    
    def forward(self, x):
        return self.network(x)

# 创建模型实例
model = SinNet(hidden_size=64)
print("模型结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 使用Adam优化器

# 4. 训练模型
num_epochs = 2000
losses = []  # 存储损失值用于绘图

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 绘制训练损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')  # 使用对数刻度更好地显示损失下降

# 6. 绘制拟合结果对比
plt.subplot(1, 2, 2)
# 绘制原始sin函数
x_plot = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
y_true = np.sin(x_plot)

# 使用模型进行预测
with torch.no_grad():
    x_tensor = torch.from_numpy(x_plot).float()
    y_pred_plot = model(x_tensor).numpy()

# 绘制图形
plt.scatter(X_numpy, y_numpy, label='Noisy data', color='blue', alpha=0.3, s=5)
plt.plot(x_plot, y_true, label='True sin(x)', color='green', linewidth=2)
plt.plot(x_plot, y_pred_plot, label='Model prediction', color='red', linewidth=2)
plt.title('Sin Function Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 评估模型在测试集上的性能
# 生成测试数据（与训练数据不同的范围，测试模型泛化能力）
X_test = np.linspace(-3.5, 3.5, 500).reshape(-1, 1)
y_test_true = np.sin(X_test)

with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_pred = model(X_test_tensor).numpy()

test_loss = np.mean((y_test_pred - y_test_true) ** 2)
print(f"测试集MSE损失: {test_loss:.6f}")

# 绘制测试结果
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test_true, label='True sin(x)', color='green', linewidth=2)
plt.plot(X_test, y_test_pred, label='Model prediction', color='red', linewidth=2)
plt.title('Model Generalization (Test on [-3.5, 3.5])')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()