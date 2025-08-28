# 作业一：调整模型层数与节点变化，对比模型loss变化
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 用于绘制损失曲线

# 1. 数据加载和预处理
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 获取所有文本数据并转换为列表
string_labels = dataset[1].tolist()  # 获取所有标签数据并转换为列表

# 2. 标签编码：将字符串标签转换为数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}  # 创建标签到索引的映射
numerical_labels = [label_to_index[label] for label in string_labels]  # 将所有标签转换为数字

# 3. 构建字符词典：为每个字符分配一个唯一的索引
char_to_index = {'<pad>': 0}  # 初始化字典，包含填充标记
for text in texts:  # 遍历所有文本
    for char in text:  # 遍历文本中的每个字符
        if char not in char_to_index:  # 如果字符不在字典中
            char_to_index[char] = len(char_to_index)  # 为字符分配新索引

index_to_char = {i: char for char, i in char_to_index.items()}  # 创建索引到字符的反向映射
vocab_size = len(char_to_index)  # 计算词典大小

max_len = 40  # 设置最大文本长度


# 4. 自定义数据集类
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts  # 存储文本数据
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为张量
        self.char_to_index = char_to_index  # 字符到索引的映射
        self.max_len = max_len  # 最大文本长度
        self.vocab_size = vocab_size  # 词典大小
        self.bow_vectors = self._create_bow_vectors()  # 预计算所有文本的词袋向量

    def _create_bow_vectors(self):
        tokenized_texts = []  # 存储所有tokenized文本
        for text in self.texts:  # 遍历所有文本
            # 将文本转换为索引序列，截取前max_len个字符
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 使用0填充到max_len长度
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)  # 添加到列表

        bow_vectors = []  # 存储所有词袋向量
        for text_indices in tokenized_texts:  # 遍历所有tokenized文本
            bow_vector = torch.zeros(self.vocab_size)  # 创建零向量
            for index in text_indices:  # 遍历文本中的每个索引
                if index != 0:  # 忽略填充标记(0)
                    bow_vector[index] += 1  # 对应位置计数加1
            bow_vectors.append(bow_vector)  # 添加到列表
        return torch.stack(bow_vectors)  # 将列表转换为张量并返回

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]  # 返回指定索引的数据和标签


# 5. 创建多个不同结构的模型
class SimpleClassifier1(nn.Module):
    """模型1: 单隐藏层，64个节点"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)  # 第一层线性变换
        out = self.relu(out)  # ReLU激活
        out = self.fc2(out)  # 第二层线性变换
        return out  # 返回输出


class SimpleClassifier2(nn.Module):
    """模型2: 双隐藏层，128和64个节点"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 输入层到第一隐藏层
        self.relu1 = nn.ReLU()  # 第一个ReLU激活函数
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第一隐藏层到第二隐藏层
        self.relu2 = nn.ReLU()  # 第二个ReLU激活函数
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # 第二隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)  # 第一层线性变换
        out = self.relu1(out)  # 第一个ReLU激活
        out = self.fc2(out)  # 第二层线性变换
        out = self.relu2(out)  # 第二个ReLU激活
        out = self.fc3(out)  # 第三层线性变换
        return out  # 返回输出


class SimpleClassifier3(nn.Module):
    """模型3: 三隐藏层，256、128和64个节点"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(SimpleClassifier3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 输入层到第一隐藏层
        self.relu1 = nn.ReLU()  # 第一个ReLU激活函数
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第一隐藏层到第二隐藏层
        self.relu2 = nn.ReLU()  # 第二个ReLU激活函数
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # 第二隐藏层到第三隐藏层
        self.relu3 = nn.ReLU()  # 第三个ReLU激活函数
        self.fc4 = nn.Linear(hidden_dim3, output_dim)  # 第三隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)  # 第一层线性变换
        out = self.relu1(out)  # 第一个ReLU激活
        out = self.fc2(out)  # 第二层线性变换
        out = self.relu2(out)  # 第二个ReLU激活
        out = self.fc3(out)  # 第三层线性变换
        out = self.relu3(out)  # 第三个ReLU激活
        out = self.fc4(out)  # 第四层线性变换
        return out  # 返回输出


# 6. 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 7. 定义模型参数
output_dim = len(label_to_index)  # 输出维度等于类别数量

# 8. 创建三个不同结构的模型
model1 = SimpleClassifier1(vocab_size, 64, output_dim)  # 单隐藏层，64个节点
model2 = SimpleClassifier2(vocab_size, 128, 64, output_dim)  # 双隐藏层，128和64个节点
model3 = SimpleClassifier3(vocab_size, 256, 128, 64, output_dim)  # 三隐藏层，256、128和64个节点

models = {
    "单隐藏层(64节点)": model1,
    "双隐藏层(128-64节点)": model2,
    "三隐藏层(256-128-64节点)": model3
}


# 9. 训练函数
def train_model(model, model_name, dataloader, num_epochs=10):
    """训练模型并返回损失历史"""
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

    loss_history = []  # 存储每个epoch的平均损失

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 累计损失

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累加损失

        epoch_loss = running_loss / len(dataloader)  # 计算平均损失
        loss_history.append(epoch_loss)  # 记录损失
        print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return loss_history  # 返回损失历史


# 10. 训练所有模型并记录损失
loss_histories = {}  # 存储每个模型的损失历史
for name, model in models.items():
    print(f"\n开始训练: {name}")
    loss_history = train_model(model, name, dataloader)
    loss_histories[name] = loss_history

# 11. 绘制损失曲线
plt.figure(figsize=(10, 6))
for name, history in loss_histories.items():
    plt.plot(history, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的训练损失对比')
plt.legend()
plt.grid(True)
plt.savefig('work1_model_comparison.png')  # 保存图像
plt.show()

# 12. 选择最佳模型进行预测
best_model_name = min(loss_histories, key=lambda k: loss_histories[k][-1])
best_model = models[best_model_name]
print(f"\n最佳模型: {best_model_name}")


# 13. 预测函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将文本转换为token序列
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充到max_len长度
    tokenized += [0] * (max_len - len(tokenized))

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:  # 忽略填充标记
            bow_vector[index] += 1

    # 添加批次维度
    bow_vector = bow_vector.unsqueeze(0)

    # 预测
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(bow_vector)  # 前向传播

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)  # 找到最大值索引
    predicted_index = predicted_index.item()  # 转换为Python标量
    predicted_label = index_to_label[predicted_index]  # 转换为标签名称

    return predicted_label


# 14. 创建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 15. 使用最佳模型进行预测
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")