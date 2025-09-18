
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def model_for_bert():
    # ---1.数据准备---
    dataset = pd.read_csv('./assets/dataset/作业数据-waimai_10k.csv', sep=',', header=0)
    print(dataset)
    print(dataset["label"])
    print(dataset["label"].values)
    print(dataset["review"])
    print(dataset["review"].values)

    # 初始化并拟合标签编码器，将文本标签转换为数字标签
    lbl = LabelEncoder()
    lbl.fit(dataset["label"].values)

    unique_labels = np.unique(dataset["label"].values)
    num_labels = len(unique_labels)
    label_data = dataset["label"].values[:500]
    review_data = dataset["review"].values[:500]

    review_train, review_test, label_train, label_test = train_test_split(
        review_data,  # 特征数据
        label_data,  # 标签数据
        train_size=0.2,  # 训练集比例
        stratify=label_data,  # 按标签分层抽样
        random_state=42  # 设置随机种子以确保可重复性
    )

    print(label_train)
    print(label_test)
    # 从预训练模型加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=num_labels)

    # 使用分词器对训练集和测试集的文本进行编码
    # truncation=True：如果文本过长则截断
    # padding=True：对齐所有序列长度，填充到最长
    # max_length=64：最大序列长度
    train_encodings = tokenizer(list(review_train), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(list(review_test), truncation=True, padding=True, max_length=64)

    # 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],           # 文本的token ID
        'attention_mask': train_encodings['attention_mask'], # 注意力掩码
        'labels': label_train.tolist() # 对应的标签
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': label_test.tolist()
    })

    # 定义用于计算评估指标的函数
    def compute_metrics(eval_pred):
        # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
        logits, labels = eval_pred
        # 找到 logits 中最大值的索引，即预测的类别
        predictions = np.argmax(logits, axis=-1)
        # 计算预测准确率并返回一个字典
        return {'accuracy': (predictions == labels).mean()}

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir='./results',              # 训练输出目录，用于保存模型和状态
        num_train_epochs=4,                  # 训练的总轮数
        per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
        per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
        warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
        weight_decay=0.01,                   # 权重衰减，用于防止过拟合
        logging_dir='./logs',                # 日志存储目录
        logging_steps=100,                   # 每隔100步记录一次日志
        eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
        save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
        load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
    )

    # 实例化 Trainer
    trainer = Trainer(
        model=model,                         # 要训练的模型
        args=training_args,                  # 训练参数
        train_dataset=train_dataset,         # 训练数据集
        eval_dataset=test_dataset,           # 评估数据集
        compute_metrics=compute_metrics,     # 用于计算评估指标的函数
    )

    # 开始训练模型
    trainer.train()
    # 在测试集上进行最终评估
    trainer.evaluate()