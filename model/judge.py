import requests
import json
from langchain_community.vectorstores import DocArrayInMemorySearch
import fitz  # PyMuPDF  # PyMuPDF
from docx import Document  # python-docx
import re
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os


# 1.保存到缓存中
def save_to_cache(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# 2.从缓存中加载
def load_from_cache(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None


# 问答API
def get_background_knowledge(question):
    question = str(question)
    system_prompt = """
    ## Role：## 你是一个直播对话风控小助手，你可以对输入的内容提取关键内容以及关键词（关键词包含产品的效果，比如医疗效果、产品、销售状况等等），请根据输入内容输出符合响应格式的内容：\n\n
    ## 响应格式：（请遵守响应格式）##
    1. 产品名称：xxx \n
    2. 产品类别：xxx、xxx \n
    3. 作用：xxx \n
    4. 关键词：xx、xxx \n
    ## 输入内容：##：{0}
    """
    user_prompt = system_prompt.format(question)
    url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions"
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDEwMjY1MSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTcyNzI1NzA3NSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTkzMDc0OTQ3NjgiLCJ1dWlkIjoiMjYyNDQ3MzAtNmViNi00ZDhmLWIyMTAtMmFjY2JhMTg0YWVmIiwiZW1haWwiOiIiLCJleHAiOjE3NDI4MDkwNzV9.MgQNBAJ0WwTwWAlvYbeom-yf2kFZoWKfheFDyNhZUc3pzY0PhYUyHYejw_jlPMVgQiAOmRMLCaDqZ2HUmLWFDw",
    }
    data = {
        "model": "internlm2.5-latest",
        "messages": [{"role": "user", "content": user_prompt}],
        "n": 1,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    response = requests.post(url, headers=header, data=json.dumps(data))
    # print(response)
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
    result = response.json()
    if "data" in result and isinstance(result["data"], list):
        # 提取嵌套的向量内容
        return result["data"][0]["embedding"]
    else:
        raise ValueError("Invalid response format: 'data' field is missing or invalid.")


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


# 读取CSV文件
def extract_ad_data_from_excel(file_path):
    df = pd.read_excel(file_path)

    ad_name_list = []
    ad_category_list = []
    ad_polic_list = []

    for index, row in df.iterrows():
        # 检查每行是否包含所需的数据
        ad_name = row["广告名称"] if pd.notnull(row["广告名称"]) else "未知"
        ad_category = row["广告类别"] if pd.notnull(row["广告类别"]) else "未知"
        ad_polic = row["涉嫌违法内容"] if pd.notnull(row["涉嫌违法内容"]) else "未知"

        ad_name_list.append(ad_name)
        ad_category_list.append(ad_category)
        ad_polic_list.append(ad_polic)

    return ad_name_list, ad_category_list, ad_polic_list


# 处理PDF
def extract_paragraphs(text, cache_dir="E:\code\zhiboJudge\model\cache"):
    """Use regular expression to extract paragraphs based on '第xx条'."""

    # 正则表达式匹配 '第xx条' 到下一个 '第xx条' 的文本
    pattern = r"第[\d一二三四五六七八九十百千万]+条\u3000[^第]*?(?=第[\d一二三四五六七八九十百千万]+[条章]\u3000|$)"  # 正则表达式

    cache_filename = os.path.join(cache_dir, "ad_paragraphs.pkl")
    paragraphs = load_from_cache(cache_filename)  # 加载缓存

    if paragraphs is None:
        paragraphs = re.findall(pattern, text)  # 匹配每条法律
        save_to_cache(paragraphs, cache_filename)  # 存储到缓存中
    return paragraphs


# 匹配Top-K个内容
def match_top_k(paragraph_vectors, query_vector, k):
    similarities = []
    for i, paragraph_vector in enumerate(paragraph_vectors):
        similarity = np.dot(paragraph_vector, query_vector) / (
            np.linalg.norm(paragraph_vector) * np.linalg.norm(query_vector)
        )
        similarities.append((i, similarity))
    # 按相似度降序排序并选出Top-K
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    # 返回对应的段落索引和相似度
    return [(index, score) for index, score in top_k]


# 遍历ad_name_list, ad_category_list, ad_polic_list内容，生成question并调用rag函数
def process_ads(ad_name_list, ad_category_list, ad_polic_list):
    question_list = []
    for ad_name, ad_category, ad_polic in zip(
        ad_name_list, ad_category_list, ad_polic_list
    ):
        if ad_name == "未知" or ad_category == "未知" or ad_polic == "未知":
            continue  # 跳过未知的数据
        question = (
            f"广告名称：{ad_name}\n广告类别：{ad_category}\n涉嫌违法内容：{ad_polic}"
        )
        print(f"Processing question:\n {question}\n")
        question_list.append(question)
        print("##" * 40)
    return question_list


# 生成最终输出的函数（示例）
def generate_output(ad_top_k_contexts, fruit_top_k_contexts, question):
    # 这里需要实现一个生成输出的逻辑，例如将匹配到的上下文和背景知识结合起来
    # 这里只是一个简单的示例，实际实现可能需要更复杂的逻辑
    system_prompt = """ ## Role ##：你是一个专业的法律博士后，传入的内容为一段处理过的直播中的对话文本，对话文本包含了主播对产品的详细介绍，包含产品种类和效果，你的任务是严格依照提供给你的法律知识对输入的对话文本进行分析，判断是否违法（0代表违法，1代表未违法）并输出具体违法了哪几条法律。\n
    ## Knowledge ##：
        - 《广告法》：{0} \n
        - 《食品安全法》：{1} \n \n
    ## Note：##
    \n注意（请遵守输出格式样式）：
    1. 一定要严格依照提供的法律。
    2. 广告中不得涉及疾病预防、治疗功能
    3. 化妆品广告不得明示或暗示产品具有医疗作用
    4. 酒类广告不得含有诱导、怂恿饮酒或宣传无节制饮酒的内容
    5. 广告中不得使用国家机关、国家机关工作人员的名义或形象
    6. 广告不得含有表示功效、安全性的断言或保证的内容
    7. 保健食品广告不得涉及疾病预防、治疗功能
    8. 广告中不得对商品的性能、功能、质量等作虚假或引人误解的商业宣传
    9. 广告中不得损害竞争对手的商业信誉、商品声誉
    \n输出格式示例：
    1. 最终结果：xx（0代表违法，1代表未违法）。 \n
    2. 违法表现：xxxx涉及xxxx 。\n
    3. 违法依据：
        - 1.《xxx》第xx条xxx。\n
        - 2....。\n
        - ... \n
    ## Input Content ##：
    """
    user_prompt = system_prompt.format(
        ad_top_k_contexts, fruit_top_k_contexts, question
    )
    result = get_background_knowledge(user_prompt)

    return result


# 主函数
def rag(question, ad_file_path,fruit_path, model_name, k=4):

    cache_dir = "E:\code\zhiboJudge\model\cache"

    # 1: 输入原始问题
    background_knowledge = get_background_knowledge(question)
    background_knowledge = background_knowledge["choices"][0]["message"]["content"]

    # 2: 读取文件内容并进行向量化处理
    if ad_file_path.endswith(".pdf") and fruit_path.endswith(".pdf"):
        ad_text = read_pdf(ad_file_path)
        fruit_text = read_pdf(fruit_path)
    elif ad_file_path.endswith(".docx") and fruit_path.endswith(".docx"):
        ad_text = read_doc(ad_file_path)
        fruit_text = read_doc(fruit_path)
    else:
        raise ValueError("Unsupported file format")

    # 3:将文本分割成段落或句子
    ad_paragraphs = extract_paragraphs(ad_text)
    fruit_paragraphs = extract_paragraphs(fruit_text)
    # for paragraph in paragraphs:
    #     print(f"{paragraph}\n")

    # 4:缓存文件路径
    ad_cache_filename_vectors = os.path.join(cache_dir, "ad_paragraph_vectors.pkl")
    fruit_cache_filename_vectors = os.path.join(cache_dir, "fruit_paragraph_vectors.pkl")
    
    ad_paragraph_vectors = load_from_cache(
        ad_cache_filename_vectors
    )  # 5:尝试从缓存中加载向量
    fruit_paragraph_vectors = load_from_cache(
        fruit_cache_filename_vectors
    )  # 尝试从缓存中加载向量
    
    # 6:如果缓存中没有向量，则进行向量化处理并保存到缓存
    if ad_paragraph_vectors is None and fruit_paragraph_vectors is None:
        ad_paragraph_vectors = [
            vectorize_text(ad_paragraph, model_name) for ad_paragraph in ad_paragraphs
        ]
        fruit_paragraph_vectors = [
            vectorize_text(fruit_paragraph, model_name) for fruit_paragraph in fruit_paragraphs
        ]
        save_to_cache(ad_paragraph_vectors, ad_cache_filename_vectors)  # 存储向量到缓存中
    # print(paragraph_vectors)

    # 7: 对背景知识进行向量化处理
    background_vector = vectorize_text(background_knowledge, model_name)

    # 8: 利用background_vector去匹配paragraph_vectors
    ad_top_k_contexts = match_top_k(ad_paragraph_vectors, background_vector, k)
    fruit_top_k_contexts = match_top_k(fruit_paragraph_vectors, background_vector, k)
    
    # 9. 遍历得到top-k个
    ad_top_k_paragraphs = [ad_paragraphs[index] for index, _ in ad_top_k_contexts]
    fruit_top_k_paragraphs = [fruit_paragraphs[index] for index, _ in fruit_top_k_contexts]
    # for i, paragraph in enumerate(top_k_paragraphs):
    #     print(f"Top {i + 1} Paragraph:\n{paragraph}\n")

    # 步骤6: 使用匹配到的上下文生成最终输出
    final_output = generate_output(ad_top_k_paragraphs, fruit_top_k_paragraphs,background_knowledge)
    final_output = final_output["choices"][0]["message"]["content"]
    return final_output


# 假设我们有一个PDF文件路径和一个问题
fl_file_path = r"E:/work/china_guanggao.pdf"
ad_input_path = r"E:/work/wf.xls"
fruit_input_path = r"E:/work/china_fruit.pdf"

ad_name_list, ad_category_list, ad_polic_list = extract_ad_data_from_excel(
    ad_input_path
)

# 遍历ad_name_list, ad_category_list, ad_polic_list内容，将这三个集合的每个元素组成一个question，question,比如"""广告名称：ad_name_list[0]\n，广告类别：ad_category_list[0]\n，直播产品介绍内容：ad_polic_list[0]\n
question_list = process_ads(ad_name_list, ad_category_list, ad_polic_list)

# 调用RAG函数
idx = 0
for question in enumerate(tqdm(question_list, desc="Processing questions")):
    idx += 1
    final_output = rag(
        question, fl_file_path, fruit_input_path, "BAAI/bge-large-zh-v1.5", k=10
    )
    print(
        f"#######################################第{idx+1}个#################################################"
    )
    print(f"\待鉴别输入：\n{str(question)}")
    print(f"\n最终鉴别结果：\n{final_output}")
    print("#" * 80)
