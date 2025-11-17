import requests
import json
import base64

def qwen_vl_ocr(image_path, api_key):
    """
    使用通义千问VL模型进行OCR文字识别
    """
    # 编码图片
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 请求头
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # 请求数据
    data = {
        "model": "qwen-vl-plus",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{image_base64}"
                        },
                        {
                            "text": "请准确识别这张图片中的所有文字，包括界面元素、按钮文字、标题、正文等，保持原有的排版格式"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }
    
    # 发送请求
    response = requests.post(
        'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        return result['output']['choices'][0]['message']['content']
    else:
        raise Exception(f"API调用失败: {response.text}")
