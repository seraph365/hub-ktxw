import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from collections import defaultdict


# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../week1/dataset.csv", sep="\t", header=None)
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

class ThreeLayerClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3= nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

hidden_dim = 128


output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
#model = ThreeLayerClassifier(vocab_size, 256, 128, output_dim)
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


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


# new_text = "帮我导航到北京"
# predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
# print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
#
# new_text_2 = "查询明天北京的天气"
# predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
# print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


# 存储不同配置的结果
results = defaultdict(list)

# 定义不同的模型配置
model_configs = [
    {"name": "128节点", "hidden_dim": 128, "layers": 2},
    {"name": "512节点", "hidden_dim": 512, "layers": 2},
    {"name": "64节点", "hidden_dim": 64, "layers": 2},
    {"name": "3层网络(256→128)", "hidden_dim1": 256, "hidden_dim2": 128, "layers": 3},
]


def create_model(config):
    if config["layers"] == 2:
        return SimpleClassifier(vocab_size, config["hidden_dim"], output_dim)
    else:
        return ThreeLayerClassifier(vocab_size, config["hidden_dim1"],
                                    config["hidden_dim2"], output_dim)


# 训练和评估每个配置
for config in model_configs:
    print(f"\n{'=' * 50}")
    print(f"训练配置: {config['name']}")
    print(f"{'=' * 50}")

    model = create_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    # 训练循环
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        results[config['name']].append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'accuracy': epoch_acc
        })

        print(f"Epoch [{epoch + 1}/10] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f}秒")

# 打印详细结果
print("\n详细结果对比:")
print("-" * 80)
for config_name, data in results.items():
    final_loss = data[-1]['loss']
    final_acc = data[-1]['accuracy']
    print(f"{config_name:20} | Final Loss: {final_loss:.4f} | Final Acc: {final_acc:.4f}")