# 作业二：构建一个sin函数，通过多层网络拟合sin函数并可视化
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 - 创建 Sin 函数数据
# 生成 1000 个在 [0, 10] 范围内的点
X_numpy = np.linspace(0, 10, 1000).reshape(-1, 1)
# 计算对应的 Sin 函数值，并添加一些噪声
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)

# 将 NumPy 数组转换为 PyTorch 张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin 函数数据生成完成。")
print(f"X 形状: {X.shape}, y 形状: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络模型
# 定义一个具有两个隐藏层的神经网络
class SinNet(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(SinNet, self).__init__()
        # 第一层：输入层到第一个隐藏层
        self.fc1 = torch.nn.Linear(input_size, hidden1_size)
        # 第二层：第一个隐藏层到第二个隐藏层
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        # 第三层：第二个隐藏层到输出层
        self.fc3 = torch.nn.Linear(hidden2_size, output_size)
        # 激活函数：使用 Tanh，因为它输出范围是 [-1, 1]，适合拟合 Sin 函数
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        # 前向传播过程
        out = self.fc1(x)  # 第一层线性变换
        out = self.activation(out)  # 应用激活函数
        out = self.fc2(out)  # 第二层线性变换
        out = self.activation(out)  # 应用激活函数
        out = self.fc3(out)  # 第三层线性变换
        return out


# 3. 创建模型实例
input_size = 1  # 输入特征维度 (只有一个 x 值)
hidden1_size = 64  # 第一个隐藏层神经元数量
hidden2_size = 32  # 第二个隐藏层神经元数量
output_size = 1  # 输出维度 (只有一个 y 值)

model = SinNet(input_size, hidden1_size, hidden2_size, output_size)

# 打印模型结构
print("神经网络模型结构:")
print(model)
print("---" * 10)

# 4. 定义损失函数和优化器
# 使用均方误差损失函数，适用于回归问题
loss_fn = torch.nn.MSELoss()

# lr=0.01 表示学习率为 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
num_epochs = 2000  # 训练轮数
losses = []  # 用于记录每轮的损失值

print("开始训练模型...")
for epoch in range(num_epochs):
    # 前向传播：通过模型计算预测值
    y_pred = model(X)

    # 计算损失：比较预测值和真实值
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度，防止梯度累积
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

    # 记录损失值
    losses.append(loss.item())

    # 每 100 轮打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)


# 7. 绘制拟合结果
# 使用训练好的模型进行预测
model.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 禁用梯度计算，节省内存
    y_predicted = model(X)  # 使用模型进行预测
    # 绘制原始数据点
    plt.scatter(X_numpy, y_numpy, label='Original Data', color='blue', alpha=0.6, s=10)
    # 绘制拟合曲线
    plt.plot(X_numpy, y_predicted.numpy(), label='Fitted Curve', color='red', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y = sin(X)')
    plt.title('Sin Function Fitting')
    plt.legend()
    plt.grid(True)

plt.tight_layout()  # 自动调整子图间距
plt.savefig('work2_fitting_graph.png')  # 保存图像
plt.show()