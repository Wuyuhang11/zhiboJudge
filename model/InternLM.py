import requests
import json

url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'
header = {
    'Content-Type':'application/json',
    "Authorization":"Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDEwMjY1MSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTcyNzI1NzA3NSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTkzMDc0OTQ3NjgiLCJ1dWlkIjoiMjYyNDQ3MzAtNmViNi00ZDhmLWIyMTAtMmFjY2JhMTg0YWVmIiwiZW1haWwiOiIiLCJleHAiOjE3NDI4MDkwNzV9.MgQNBAJ0WwTwWAlvYbeom-yf2kFZoWKfheFDyNhZUc3pzY0PhYUyHYejw_jlPMVgQiAOmRMLCaDqZ2HUmLWFDw"
}
data = {
    "model": "internlm2.5-latest",  
    "messages": [{
        "role": "user",
        "content": """ """
    }],
    "n": 1,
    "temperature": 0.8,
    "top_p": 0.9
}

res = requests.post(url, headers=header, data=json.dumps(data))
print(res.status_code)
print(res.json())