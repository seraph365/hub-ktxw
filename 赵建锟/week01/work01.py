from sklearn import linear_model #线性模型
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import jieba

data = pd.read_csv("dataset.csv",sep="\t",header=None,names=["text","label"])
texts = data["text"].values
labels = data["label"].values

def chinese_cut(text):
    return " ".join(jieba.cut(text))

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform([chinese_cut(text) for text in x_train])
x_test_tfidf = vectorizer.transform([chinese_cut(text) for text in x_test])

model = LogisticRegression(max_iter=10000)
model.fit(x_train_tfidf, y_train)
predictions = model.predict(x_test_tfidf)
result = predictions == y_test
print(result)
print(np.sum(result==True) / len(predictions))

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_tfidf, y_train)
predictions = model.predict(x_test_tfidf)
result = predictions == y_test
print(result)
print(np.sum(result==True) / len(predictions))
