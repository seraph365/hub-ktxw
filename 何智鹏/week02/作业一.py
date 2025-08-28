import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 数据加载和预处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist() # 样本集
string_labels = dataset[1].tolist() # 标签

# 将标签按数字分类
label_to_index = {label: i for i, label in enumerate(set(string_labels))}

# 将原数据标签换成数字
numerical_labels = [label_to_index[label] for label in string_labels]

# 将数据每个字按数字分类
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

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
class SimpleClassifier1(nn.Module):  # 原始结构
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SimpleClassifier2(nn.Module):  # 增加层数
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class SimpleClassifier3(nn.Module):  # 增加节点数
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)  # 双倍节点
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)  # 保持原有节点
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class SimpleClassifier4(nn.Module):  # 减少层数和节点数
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier4, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim // 2)  # 一半节点
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 训练函数
def train_model(model, dataloader, num_epochs=10, lr=0.04):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
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
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return losses

# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 模型参数
input_dim = vocab_size
hidden_dim = 128
output_dim = len(label_to_index)

# 训练不同结构的模型并记录loss
print("=" * 50)
print("训练原始模型 (2层，128节点)")
model1 = SimpleClassifier1(input_dim, hidden_dim, output_dim)
losses1 = train_model(model1, dataloader)

print("=" * 50)
print("训练增加层数的模型 (3层，128节点)")
model2 = SimpleClassifier2(input_dim, 128, 64, output_dim)
losses2 = train_model(model2, dataloader)

print("=" * 50)
print("训练增加节点数的模型 (3层，256节点)")
model3 = SimpleClassifier3(input_dim, hidden_dim, output_dim)
losses3 = train_model(model3, dataloader)

print("=" * 50)
print("训练减少层数和节点数的模型 (2层，64节点)")
model4 = SimpleClassifier4(input_dim, hidden_dim, output_dim)
losses4 = train_model(model4, dataloader)

# 绘制loss曲线
plt.figure(figsize=(12, 8))
plt.plot(losses1, label='原始模型 (2层, 128节点)', linewidth=2)
plt.plot(losses2, label='增加层数 (3层, 128节点)', linewidth=2)
plt.plot(losses3, label='增加节点数 (3层, 256节点)', linewidth=2)
plt.plot(losses4, label='减少层数和节点数 (2层, 64节点)', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的Loss对比')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison.png')
plt.show()

# 打印最终loss对比
print("\n" + "=" * 60)
print("最终Loss对比:")
print(f"原始模型 (2层, 128节点): {losses1[-1]:.4f}")
print(f"增加层数 (3层, 128节点): {losses2[-1]:.4f}")
print(f"增加节点数 (3层, 256节点): {losses3[-1]:.4f}")
print(f"减少层数和节点数 (2层, 64节点): {losses4[-1]:.4f}")

# 测试函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

# 测试所有模型
test_texts = ["帮我导航到北京", "查询明天北京的天气", "播放周杰伦的音乐", "打开空调"]

print("\n" + "=" * 60)
print("模型预测结果对比:")

for text in test_texts:
    print(f"\n输入文本: '{text}'")
    pred1 = classify_text(text, model1, char_to_index, vocab_size, max_len, index_to_label)
    pred2 = classify_text(text, model2, char_to_index, vocab_size, max_len, index_to_label)
    pred3 = classify_text(text, model3, char_to_index, vocab_size, max_len, index_to_label)
    pred4 = classify_text(text, model4, char_to_index, vocab_size, max_len, index_to_label)
    
    print(f"  原始模型: {pred1}")
    print(f"  增加层数: {pred2}")
    print(f"  增加节点: {pred3}")
    print(f"  减少参数: {pred4}")