import re
import pandas as pd

# 读取日志文件内容
with open("result.log", "r", encoding="utf-8") as file:
    content = file.read()

# 定义正则表达式，匹配每个块的完整内容
block_pattern = r"#######################################第\d+个#################################################(.*?)################################################################################"
blocks = re.findall(block_pattern, content, re.S)

# 定义字段名称
columns = ["广告名称", "广告类别", "涉嫌违法内容", "最终结果", "违法表现", "违法依据"]
data = []

# 遍历每个块，提取字段
for block in blocks:
    # 提取每个字段
    name = re.search(r"广告名称：(.*?)\n", block)
    category = re.search(r"广告类别：(.*?)\n", block)
    illegal_content = re.search(r"涉嫌违法内容：(.*?)最终鉴别结果：", block, re.S)
    final_result = re.search(r"最终鉴别结果：\n1\. 最终结果：(.*?)\n", block, re.S)
    illegal_behavior = re.search(r"2\. 违法表现：(.*?)\n\n3\. 违法依据：", block, re.S)
    legal_basis = re.search(r"3\. 违法依据：(.*?)当前违法数量为：", block, re.S)

    # 将字段数据添加到列表，清理空白和换行符
    data.append([
        name.group(1).strip() if name else "",
        category.group(1).strip() if category else "",
        illegal_content.group(1).strip().replace("\n", " ") if illegal_content else "",
        final_result.group(1).strip() if final_result else "",
        illegal_behavior.group(1).strip().replace("\n", " ") if illegal_behavior else "",
        legal_basis.group(1).strip().replace("\n", " ") if legal_basis else ""
    ])

# 将数据转换为 DataFrame
df = pd.DataFrame(data, columns=columns)

# 保存到 Excel 文件
output_path = "广告分析结果_按块解析.xlsx"
df.to_excel(output_path, index=False)

print(f"解析完成，共提取 {len(blocks)} 个块，结果已保存到 {output_path}")
