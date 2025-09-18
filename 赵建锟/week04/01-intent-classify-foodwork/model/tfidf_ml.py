from typing import Union, List

import jieba
import pandas as pd
from joblib import load
from config import TFIDF_MODEL_PKL_PATH

# 加载模型
tfidf, model = load(TFIDF_MODEL_PKL_PATH)

# 停用词
cn_stopwords = pd.read_csv('http://mirror.coggle.club/stopwords/baidu_stopwords.txt', header=None)[0].values


def model_for_tfidf(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None

    if isinstance(request_text, str):
        query_words = " ".join([x for x in jieba.lcut(request_text) if x not in cn_stopwords])
        classify_result = list(model.predict(tfidf.transform([query_words])))
    elif isinstance(request_text, list):
        query_words = []
        for text in request_text:
            query_words.append(
                " ".join([x for x in jieba.lcut(text) if x not in cn_stopwords])
            )
            classify_result = list(model.predict(tfidf.transform(query_words)))
    else:
        raise Exception("格式不支持")

    return classify_result
