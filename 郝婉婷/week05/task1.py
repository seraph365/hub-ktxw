import openai

# 设置 DeepSeek API Key 和 Base URL
openai.api_key = "你的_API_Key"
openai.api_base = "https://api.deepseek.com"

# 调用聊天补全接口
response = openai.ChatCompletion.create(
    model="deepseek-chat",  # 或其他 DeepSeek 模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好！请介绍一下你自己。"},
    ],
    stream=False  # 是否流式输出
)

# 打印回复
print(response.choices[0].message.content)
