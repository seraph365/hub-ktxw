import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X_numpy = np.random.rand(100, 1) * 10
y_numpy = 2 * np.sin(X_numpy) + 1 + 0.5 * np.random.randn(100, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"数据形状: X={X.shape}, y={y.shape}")
print("---" * 10)


class SinApproximator(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super(SinApproximator, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


model = SinApproximator(hidden_size=64)
print("模型结构:")
print(model)
print("---" * 10)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 2000
losses = []

print("开始训练...")
for epoch in range(num_epochs):

    y_pred = model(X)

    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("训练完成！")
print("---" * 10)


plt.figure(figsize=(12, 5))

X_test = np.linspace(0, 10, 300).reshape(-1, 1)  #按顺序做预测点，否则连线时杂乱无章的（直接将X_numpy排序作为预测点应该也同样合理）
X_test_tensor = torch.from_numpy(X_test).float()

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6, s=20)
# 真实函数
plt.plot(X_test, 2 * np.sin(X_test) + 1, 'g--', label='True function: 2*sin(x)+1', linewidth=2)
# 模型预测函数
plt.plot(X_test, y_test_pred.numpy(), 'r-', label='Model prediction', linewidth=2)

plt.xlabel('X')
plt.ylabel('y')
plt.title('Function Approximation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
