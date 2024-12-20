import requests

url = "https://api.siliconflow.cn/v1/rerank"

payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Apple",
    "documents": ["苹果", "香蕉", "水果", "蔬菜"],
    "top_n": 4,
    "return_documents": False,
    "max_chunks_per_doc": 1024,
    "overlap_tokens": 80
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)