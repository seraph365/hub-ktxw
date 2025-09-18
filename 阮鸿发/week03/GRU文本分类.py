import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# 数据集第一列 文本
texts = dataset[0].tolist()
# 数据集第二列 类别
string_labels = dataset[1].tolist()

#set(string_labels)创建一个集合，提取唯一标签，enumerate用于同时获取对象索引和元素{'A'}-》[(0,'A')] 在转换为['A',0]
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
#[0,1,2]
numerical_labels = [label_to_index[label] for label in string_labels]

#初始化一个字典，将'<pad>'映射到索引0
char_to_index = {'<pad>': 0}
#遍历每个文本
for text in texts:
    # 遍历每个文本中每个字符
    for char in text:
        # 检查字符是否在字典中
        if char not in char_to_index:
            # 为新字符配与字典等长的索引
            char_to_index[char] = len(char_to_index)

#从索引到字符的反向映射字典
index_to_char = {i: char for char, i in char_to_index.items()}
#词汇表大小
vocab_size = len(char_to_index)

# 取文本的前40个字符
max_len = 40

class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW GRU Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
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
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
