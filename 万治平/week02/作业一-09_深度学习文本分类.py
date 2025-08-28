import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os

# 加载数据文件（使用脚本所在目录的相对路径，避免运行目录差异影响）
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset.csv")
dataset = pd.read_csv(dataset_path, sep="\t", header=None)


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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model: nn.Module, dataloader: DataLoader, num_epochs: int = 10, lr: float = 0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"  Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_epoch_loss:.4f}")
    return epoch_losses


# 构建数据集与数据加载器
set_seed(42)
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 对比不同网络深度与宽度的训练损失
output_dim = len(label_to_index)
num_epochs = 10
learning_rate = 0.01

configs = [
    ("1x64", [64]),
    ("2x64", [64, 64]),
    ("3x64", [64, 64, 64]),
    ("1x128", [128]),
    ("2x128", [128, 128]),
    ("3x128", [128, 128, 128]),
]

learning_rate_config = {
    "1x64": 0.01,
    "2x64": 0.01,
    "3x64": 0.01,
    "1x128": 0.005,
    "2x128": 0.01,
    "3x128": 0.01,
}

results = {}
for name, hidden_dims in configs:
    print(f"\n=== 训练配置: {name} (隐藏层: {hidden_dims}) ===")
    set_seed(42)  # 确保每次对比具有可重复性
    model = MLPClassifier(vocab_size, hidden_dims, output_dim)
    losses = train_model(model, dataloader, num_epochs=num_epochs, lr=learning_rate_config[name])
    results[name] = losses

print("\n=== 各配置最终 Loss 对比 ===")
summary = sorted(((name, losses[-1]) for name, losses in results.items()), key=lambda x: x[1])
for name, final_loss in summary:
    print(f"配置 {name}: 最终 Loss = {final_loss:.4f}")

print("\n=== 每个配置的逐 Epoch Loss ===")
for name, losses in results.items():
    pretty_losses = ", ".join(f"{v:.4f}" for v in losses)
    print(f"{name}: [" + pretty_losses + "]")


# def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
#     tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
#     tokenized += [0] * (max_len - len(tokenized))

#     bow_vector = torch.zeros(vocab_size)
#     for index in tokenized:
#         if index != 0:
#             bow_vector[index] += 1

#     bow_vector = bow_vector.unsqueeze(0)

#     model.eval()
#     with torch.no_grad():
#         output = model(bow_vector)

#     _, predicted_index = torch.max(output, 1)
#     predicted_index = predicted_index.item()
#     predicted_label = index_to_label[predicted_index]

#     return predicted_label


# index_to_label = {i: label for label, i in label_to_index.items()}

# new_text = "帮我导航到北京"
# predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
# print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# new_text_2 = "查询明天北京的天气"
# predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
# print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")