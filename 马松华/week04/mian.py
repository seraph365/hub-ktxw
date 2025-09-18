# python自带库
import time
import traceback
from typing import Union

# 第三方库
import openai
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyResponse
from data_schema import TextClassifyRequest
from model.prompt import model_for_gpt
from model.bert import model_for_bert
from model.regex_rule import model_for_regex
from model.tfidf_ml import model_for_tfidf
from logger import logger
from model.week_bert import model_for_bert_week

app = FastAPI()

# 新增week作业
@app.post("/v1/text-cls/week_bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用TFIDF进行文本分类

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    # info 日志
    try:
        response.classify_result = model_for_bert_week(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response

