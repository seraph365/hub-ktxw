import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""
作业1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
    分别使用 1层网络32个节点，1层网络128个节点，1层网络512个节点，2层网络128、32个节点，2层网络512、32个节点，3层网络512、128、32个节点 模型
    进行训练，记录并绘制上述各个模型的loss变化
"""


def init_data():
    """
    从文件加载数据，并对数据进行处理
    :return: 文本集、标签索引、数字化标签、字典
    """
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

    return texts, label_to_index, numerical_labels, char_to_index


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


class ClassifierWith1Hidden(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(ClassifierWith1Hidden, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ClassifierWith2Hidden(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(ClassifierWith2Hidden, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class ClassifierWith3Hidden(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(ClassifierWith3Hidden, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()

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


def train_data(dataloader, model):
    criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # epoch： 将数据集整体迭代训练一次
    # batch： 数据集汇总为一批训练一次
    loss_list: list[float] = []
    num_epochs = 10
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 50 == 0:
            #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        loss_ = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_ :.4f}")
        loss_list.append(loss_)

    return loss_list


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
    predicted_label = index_to_label[predicted_index.item()]

    return predicted_label


def plot_results(X: list, y: dict) -> None:
    plt.rcParams['font.family'] = 'SimHei'  # 设置中文的字体
    for model_name, loss_list in y.items():
        plt.plot(X, loss_list, linestyle='-', label=model_name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    texts, label_to_index, numerical_labels, char_to_index = init_data()
    vocab_size = len(char_to_index)
    index_to_label = {i: label for label, i in label_to_index.items()}
    max_len = 40
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据
    output_dim = len(label_to_index)
    model_dict = dict()
    model_32 = ClassifierWith1Hidden(vocab_size, 32, output_dim)  # 1层网络32个节点
    model_dict['1层网络32个节点'] = model_32
    model_128 = ClassifierWith1Hidden(vocab_size, 128, output_dim)  # 1层网络128个节点
    model_dict['1层网络128个节点'] = model_128
    model_512 = ClassifierWith1Hidden(vocab_size, 512, output_dim)  # 1层网络512个节点
    model_dict['1层网络512个节点'] = model_512
    model_128_32 = ClassifierWith2Hidden(vocab_size, 128, 32, output_dim)  # 2层网络128、32个节点
    model_dict['2层网络128、32个节点'] = model_128_32
    model_512_32 = ClassifierWith2Hidden(vocab_size, 512, 32, output_dim)  # 2层网络512、32个节点
    model_dict['2层网络512、32个节点'] = model_512_32
    model_512_128_32 = ClassifierWith3Hidden(vocab_size, 512, 128, 32, output_dim)  # 3层网络512、128、32个节点
    model_dict['3层网络512、128、32个节点'] = model_512_128_32
    print(model_dict.keys())

    model_loss = dict()
    for model_name, model in model_dict.items():
        print('-' * 50)
        print(f"使用 {model_name} 进行训练和预测：")
        loss_list = train_data(dataloader, model)
        model_loss[model_name] = loss_list
        new_text = "帮我导航到北京"
        predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
        print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
        new_text_2 = "查询明天北京的天气"
        predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
        print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
        print('-' * 50 + "\n")

    # 绘制不同层数不同节点数网络的loss情况
    plot_results(list(range(10)), model_loss)


if __name__ == '__main__':
    main()
