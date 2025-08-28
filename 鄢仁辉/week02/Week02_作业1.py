# 作业：调整 09_深度学习文本分类.py 代码中模型的层数和节点数，对比模型的loss变化

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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


class OneLayerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(OneLayerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TwoLayerModel(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
        super(TwoLayerModel, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_dim 到 hidden_size1
            nn.Linear(input_dim, hidden_size1),
            nn.ReLU(), # 增加模型的复杂度，非线性

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            # 输出层：从 hidden_size2 到 output_dim
            nn.Linear(hidden_size2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

model1 = OneLayerModel(vocab_size, hidden_dim=128, output_dim=len(label_to_index)) # 维度和精度有什么关系？
model2 = OneLayerModel(vocab_size, hidden_dim=64, output_dim=len(label_to_index))
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model1.parameters(), lr=0.01)
optimizer = optim.SGD(model2.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10
sumLoss = 0
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model1.train()
    running_loss = 0.0
    hidden_dim = 128
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if idx % 50 == 0:
        #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    loss = running_loss / len(dataloader)
    sumLoss += loss
aveLoss = sumLoss / num_epochs
print(f'One Layer Model: hidden dimension = {hidden_dim}, loss = {aveLoss:.4f}')

num_epochs = 10
sumLoss = 0
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model2.train()
    running_loss = 0.0
    hidden_dim = 64
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if idx % 50 == 0:
        #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    loss = running_loss / len(dataloader)
    sumLoss += loss
aveLoss = sumLoss / num_epochs
print(f'One Layer Model: hidden dimension = {hidden_dim}, loss = {aveLoss:.4f}')

model3 = TwoLayerModel(vocab_size, hidden_size1=32, hidden_size2=32, output_dim=len(label_to_index)) # 维度和精度有什么关系？
model4 = TwoLayerModel(vocab_size, hidden_size1=16, hidden_size2=16, output_dim=len(label_to_index))
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model3.parameters(), lr=0.01)
optimizer = optim.SGD(model4.parameters(), lr=0.01)

num_epochs = 10
sumLoss = 0
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model3.train()
    running_loss = 0.0
    hidden_dim = 32
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model3(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if idx % 50 == 0:
        #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    loss = running_loss / len(dataloader)
    sumLoss += loss
aveLoss = sumLoss / num_epochs
print(f'Two Layer Model: hidden dimension = {hidden_dim}, loss = {aveLoss:.4f}')

num_epochs = 10
sumLoss = 0
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model4.train()
    running_loss = 0.0
    hidden_dim = 16
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model4(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if idx % 50 == 0:
        #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    loss = running_loss / len(dataloader)
    sumLoss += loss
aveLoss = sumLoss / num_epochs
print(f'Two Layer Model: hidden dimension = {hidden_dim}, loss = {aveLoss:.4f}')
