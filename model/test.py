import requests
import json
from langchain_community.vectorstores import DocArrayInMemorySearch
import fitz  # PyMuPDF  # PyMuPDF
from docx import Document  # python-docx
import re

# 问答API
def get_background_knowledge(question):
    system_prompt = "你是一个背景思考小助手，你可以根据输入的内容识别出主人公做了一件什么样的事情：\n\n"
    url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions"
    header = {
        "Content-Type": "application/json",
        "Authorization":"Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDEwMjY1MSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTcyNzI1NzA3NSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTkzMDc0OTQ3NjgiLCJ1dWlkIjoiMjYyNDQ3MzAtNmViNi00ZDhmLWIyMTAtMmFjY2JhMTg0YWVmIiwiZW1haWwiOiIiLCJleHAiOjE3NDI4MDkwNzV9.MgQNBAJ0WwTwWAlvYbeom-yf2kFZoWKfheFDyNhZUc3pzY0PhYUyHYejw_jlPMVgQiAOmRMLCaDqZ2HUmLWFDw",
    }
    data = {
        "model": "internlm2.5-latest",
        "messages": [{"role": "user", "content": system_prompt + question}],
        "n": 1,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    response = requests.post(url, headers=header, data=json.dumps(data))
    print(response)
    return response.json()


# 向量化处理
def vectorize_text(text, model_name):
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {"model": model_name, "input": text, "encoding_format": "float"}
    headers = {
        "Authorization": "Bearer sk-kifqepgmrmlstabhxlmxrylkppvcppumtvwcdbwajgotuvvk",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()["data"]


# 读取PDF文件
def read_pdf(file_path):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 读取Word文档
def read_doc(file_path):
    """遍历文档中的所有段落."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_paragraphs(text):
    """Use regular expression to extract paragraphs based on '第xx条'."""
    # 正则表达式匹配 '第xx条' 到下一个 '第xx条' 的文本
    pattern = r'第[\d一二三四五六七八九十百千万]+条\u3000[^第]*?(?=第[\d一二三四五六七八九十百千万]+[条章]\u3000|$)'
    paragraphs = re.findall(pattern, text)
    return paragraphs

# 匹配Top-K个内容
def match_top_k(contexts, query_vector, k):
    # 使用DocArrayInMemorySearch进行向量匹配
    search = DocArrayInMemorySearch(contexts)
    top_k_results = search.search(query_vector, k)
    return top_k_results


# 主函数
def rag(question, file_path, model_name, k=4):
    # 步骤1: 输入原始问题
    background_knowledge = get_background_knowledge(question)

    # 步骤2: 读取文件内容并进行向量化处理
    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = read_doc(file_path)
    else:
        raise ValueError("Unsupported file format")

    # 将文本分割成段落或句子
    paragraphs = extract_paragraphs(text)
    paragraph_vectors = [
        vectorize_text(paragraph, model_name) for paragraph in paragraphs
    ]

    # 步骤3: 对背景知识进行向量化处理
    background_vector = vectorize_text(
        background_knowledge["choices"][0]["message"]["content"], model_name
    )

    # 步骤4: 分词处理（假设背景知识已经是分词后的）
    background_tokens = background_knowledge["choices"][0]["message"]["content"].split()

    # 步骤5: 匹配得到Top-K个内容作为提示词上下文
    top_k_contexts = match_top_k(paragraph_vectors, background_vector, k)

    # 步骤6: 使用匹配到的上下文生成最终输出
    # 这里需要一个生成函数，它将使用匹配到的上下文和背景知识来生成最终的输出
    # 假设我们有一个函数generate_output，它接受上下文和背景知识作为参数
    final_output = generate_output(top_k_contexts, background_tokens)

    return final_output


# 假设我们有一个PDF文件路径和一个问题
file_path = r"E:/work/china_guanggao.pdf"
question = "00：57：31然后她又买了又坚持吃，她跟我说吃完特别有用，她老公先发现的，说她的脸就好像回到了21岁她们俩刚认识的时候，然后说她全身也特别紧，全身上下紧紧的，我说真的假的，他送了我一瓶，我吃了，我跟你说太有用了，真的。其实吃第一瓶的时候我还没有觉得那么明显，因为吃那个时候是二二年嘛，你们有没有人跟我，呃，二二年看过我直播的有不？因为二二年的时候我是晚上直播到半夜两点半，直播特别久，播到两点半，然后到家，因为我是在公司直播，到家是三点，洗完澡呀，吃个饭呀，玩一会儿啊，都早上六点才睡觉，12点起来又直播，你想我34岁，每天早上六点睡觉，就睡六个小时，我脸特黄，就是那个反重力胶囊嘛，咱们这不有吃过的吗？然后呢，我吃完第一瓶，我就觉得还好吧，就好像心理心理作用啊，脸没那么黄了。01：00：10然后但是我吃完第二瓶，我一点不夸张，每一个进来我直播间都问我是不是镀了什么幼胎脸了，度了什么轮廓固定了，度了什么什么什么少女什么玩意儿的，说我的脸变化巨大。"

# 调用RAG函数
final_output = rag(question, file_path, "BAAI/bge-large-zh-v1.5", k=5)
print(final_output)


# 生成最终输出的函数（示例）
def generate_output(top_k_contexts, background_tokens):
    # 这里需要实现一个生成输出的逻辑，例如将匹配到的上下文和背景知识结合起来
    # 这里只是一个简单的示例，实际实现可能需要更复杂的逻辑
    output = (
        "Top-K contexts: "
        + str(top_k_contexts)
        + "\nBackground knowledge: "
        + " ".join(background_tokens)
    )
    return output
