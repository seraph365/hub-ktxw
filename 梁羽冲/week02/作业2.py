import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 生成模拟数据 
X = torch.rand(100,1,dtype=torch.float32,device=device) * 10
Y = torch.sin(X)

class multiNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, hidden_layers_num, output_size) -> None:
        super(multiNetwork,self).__init__()
        self.hidden_layers_num = hidden_layers_num
        layers = []
        
        layers.append(nn.Linear(input_size,hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(self.hidden_layers_num):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim,output_size))
        self.model = nn.Sequential(*layers)
            
    def forward(self,x):
        out = self.model(x)
        return out
    
    
input_size= len(X[0])
hidden_dim = 256
hidden_layers_num = 3
output_size = len(Y[0])
num_epochs = 10000
lr = 0.0005
model = multiNetwork(input_size, hidden_dim, hidden_layers_num, output_size).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr)

for epoch in range(num_epochs):
    model.train()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
model.eval()
with torch.no_grad():
    Y_pred = model(X)
  
    X_test = torch.linspace(0, 10, 1000, dtype=torch.float32, device=device).reshape(-1, 1)
    Y_test_pred = model(X_test)

X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()
Y_pred_np = Y_pred.cpu().numpy()
X_test_np = X_test.cpu().numpy()
Y_test_pred_np = Y_test_pred.cpu().numpy()

# %matplotlib inline
plt.figure(figsize=(10, 6))
plt.scatter(X_np, Y_np, c='blue', label='Raw data', alpha=0.6)
plt.plot(X_test_np, Y_test_pred_np, 'r-', label='Predictive data', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('A multilayer neural network is used to fit a sine function.')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
