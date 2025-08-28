import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 数据加载和预处理
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 定义不同结构的模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()
        layers = []

        # 创建隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return losses


# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义不同模型结构
output_dim = len(label_to_index)
num_epochs = 10

# 模型配置列表：每个元素是(隐藏层维度列表, 学习率, 模型名称)
model_configs = [
    ([128], 0.01, "1层128节点"),  # 原始结构
    ([256], 0.01, "1层256节点"),  # 增加节点数
    ([512], 0.01, "1层512节点"),  # 进一步增加节点数
    ([128, 64], 0.01, "2层128-64节点"),  # 增加层数
    ([256, 128], 0.01, "2层256-128节点"),  # 增加层数和节点数
    ([128, 128, 64], 0.01, "3层128-128-64节点"),  # 更多层
]

# 存储每个模型的损失历史
all_losses = {}

# 训练所有模型配置
for hidden_dims, lr, model_name in model_configs:
    print(f"\n训练模型: {model_name}")
    model = SimpleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = train_model(model, dataloader, criterion, optimizer, num_epochs, model_name)
    all_losses[model_name] = losses

# 绘制损失曲线
plt.figure(figsize=(12, 8))
for model_name, losses in all_losses.items():
    plt.plot(range(1, num_epochs + 1), losses, label=model_name, marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的Loss变化对比')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison.png')
plt.show()

# 打印最终损失对比
print("\n各模型最终损失对比:")
for model_name, losses in all_losses.items():
    print(f"{model_name}: {losses[-1]:.4f}")
