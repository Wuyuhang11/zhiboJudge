import requests

url = "https://api.siliconflow.cn/v1/embeddings"

payload = {
    "model": "BAAI/bge-large-zh-v1.5",
    "input": "硅基流动embedding上线，多快好省的 embedding 服务，快来试试吧",
    "encoding_format": "float"
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)