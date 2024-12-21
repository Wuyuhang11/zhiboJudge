import pandas as pd

def extract_ad_data_from_excel(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 初始化列表
    ad_name_list = []
    ad_category_list = []
    ad_polic_list = []
    
    # 遍历DataFrame，提取所需信息
    for index, row in df.iterrows():
        ad_name_list.append(row['广告名称'])
        ad_category_list.append(row['广告类别'])
        ad_polic_list.append(row['涉嫌违法内容'])
    
    # 返回提取的列表
    return ad_name_list, ad_category_list, ad_polic_list

def main():
    # 指定Excel文件路径
    file_path = 'E:/work/wf.xls'
    
    # 调用函数并获取结果
    ad_names, ad_categories, ad_policies = extract_ad_data_from_excel(file_path)
    
    # 打印结果
    print("广告名称列表:")
    for name in ad_names:
        print(name)
    
    print("\n广告类别列表:")
    for category in ad_categories:
        print(category)
    
    print("\n涉嫌违法内容列表:")
    for policy in ad_policies:
        print(policy)

if __name__ == "__main__":
    main()