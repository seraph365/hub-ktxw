import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt


# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(10000, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 10000 个在 [0, 1) 范围内均匀分布的随机浮点数。

Y_numpy = 2*np.sin(1.5*X_numpy+0.5)+1 + np.random.randn(10000, 1)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
Y = torch.from_numpy(Y_numpy).float()

print("数据生成完成。")
print("---" * 10)

class sin_function_fitting(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(sin_function_fitting, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

input_dim =X.shape[1]
hidden_dim = 128
output_dim =Y.shape[1]
model = sin_function_fitting(input_dim, hidden_dim, output_dim) # 维度和精度有什么关系？model=SimpleClassifier(1, 10, 1)
criterion = nn.MSELoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 100
train_losses = []  # 用于存储每个epoch的损失
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    # 计算平均损失并记录
    avg_loss = total_loss / num_batches
    train_losses.append(avg_loss)

    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')
print("\n训练完成！")

# 5. 可视化损失曲线
plt.figure(figsize=(12, 5))

# 子图1：损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# 子图2：模型预测结果
plt.subplot(1, 2, 2)
model.eval()
with torch.no_grad():
    # 使用部分数据进行可视化
    sample_indices = torch.randperm(X.size(0))[:200]
    X_sample = X[sample_indices]
    Y_sample = Y[sample_indices]
    y_predicted = model(X_sample)

plt.scatter(X_sample.numpy(), Y_sample.numpy(), label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_sample.numpy(), y_predicted.numpy(), label='Predicted data', color='red', alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Model Prediction')

plt.tight_layout()
plt.show()


