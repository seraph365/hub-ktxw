import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# 1. 构造训练数据集 sin函数，需要x和y

# 定义训练集样本数
# 创建一个等差数列，从-2π到2π，共1000个点
train_x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
# 改变x的形状，从一维数组变为二维数组，-1表示自动计算行数，1表示列数为1，转换为1列后，表示1000行，1列，1个特征
train_x = train_x.reshape(-1, 1)

# 根据x和sin函数计算y，y=sin(x),y也是二维数组，1000行，1列
train_y = np.sin(train_x)
# 将y值增加一些噪声（可方便让预测值和实际值存在一些差异，方便图形化看出区别）
train_y = train_y + np.random.randn(1000, 1) * 0.05

# 将train_x和train_y转换为tensor
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()

print("训练数据集准备完毕")
print("训练数据集形状：", train_x.shape, train_y.shape)
print("---" * 10)

# 2. 定义多层全链接网络，用于学习sin函数
# 定义一个多层全链接神经网络，包含3个隐藏层，每个隐藏层有100个神经元
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        # 定义一个全链接层，输入特征数为1，输出特征数为100
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义一个全链接层，输入特征数为100，输出特征数为100
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 定义一个全链接层，输入特征数为100，输出特征数为1
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 定义一个激活函数，ReLU，将x通过全链接层，再通过ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 将x通过全链接层，再通过ReLU激活函数
        x = torch.relu(self.fc2(x))
        # 将x通过全链接层，再通过ReLU激活函数
        x = self.fc3(x)
        # 返回x
        return x

# 定义一个网络，输入特征数为1，输出特征数为1，隐藏层特征数为100
net = Net(input_size=1, hidden_size=100, output_size=1)
# 定义损失函数，使用均方误差损失函数
criterion = nn.MSELoss()
# 定义优化器，使用Adam优化器，学习率为0.01
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


# 3. 训练模型
for epoch in range(1000):
    # 训练模式
    net.train()
    # 清空梯度
    optimizer.zero_grad()
    # 前向传播
    pred_y = net(train_x)
    # 计算损失
    loss = criterion(pred_y, train_y)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss: {loss.item()}")

print("模型训练完成")
print("---" * 10)

# 4. 测试模型
# 验证测试模式
net.eval()

# 测试集从-2π到2π，共100个点
test_x = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
# 将test_x转换为tensor
test_x = torch.from_numpy(test_x).float()

# 验证测试模式不计算梯度
with torch.no_grad():
    # 模型预测
    pred_y = net(test_x).detach().numpy()

# 绘制预测值和实际值的图像
plt.figure(figsize=(10, 5))
plt.plot(test_x, pred_y, label='Predicted', color='red', linewidth=2)
plt.plot(test_x, np.sin(test_x), label='Actual', color='blue', linewidth=2)
plt.title('sin函数 Predicted vs Actual', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('predicted_vs_actual.png', dpi=300)
plt.show()
