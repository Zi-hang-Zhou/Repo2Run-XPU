import requests
import json
import os
import openai # 尽管我们不用它，但其他文件 import 了它，所以保留
import time   # 确保 time 库被导入

def get_llm_response(model: str, messages, temperature = 0.0, n = 1, max_tokens = 4096):
    """
    使用 'requests' 库（而不是 'openai' 库）来调用 API。
    绕过 'openai==0.28.0' 库中处理 http api_base 的 Bug。
    """


    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")

    if not key or not base_url:
        print("错误：OPENAI_API_KEY 或 OPENAI_API_BASE 环境变量未设置！")
        return [None], {"total_tokens": 0}

    url = f"{base_url}/chat/completions"


    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    payload_str = json.dumps(payload)

    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
    }


    max_retry = 5
    count = 0
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload_str, timeout=300)
            response.raise_for_status() 
            response_json = response.json()

            if "error" in response_json:
                print(f"API 代理返回错误: {response_json['error']}")
                raise Exception(response_json['error'])

            result_text = response_json["choices"][0]["message"]["content"]
            usage_data = response_json.get("usage", {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0})

            return [result_text], usage_data

        except Exception as e:
            count = count + 1
            print(f"LLM API 调用失败 (第 {count} 次尝试): {e}")

            try:
                if response:
                    print(f"失败时的响应内容: {response.text[:500]}...")
            except:
                pass

            if count > max_retry:
                print(f"达到最大重试次数 ({max_retry})，彻底失败。")
                return [None], {"total_tokens": 0}

            time.sleep(3)