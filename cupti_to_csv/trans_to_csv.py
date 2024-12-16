import json
import csv
import os
from collections import defaultdict

# 定义保存CSV文件的主目录
output_directory = 'trans_to_csv'
log_directory = 'log'

# 检查并创建主目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 找出 log 目录下所有 JSON 文件
json_files = [f for f in os.listdir(log_directory) if f.endswith('.json')]

# 遍历每个 JSON 文件
for json_file_name in json_files:
    # 读取 JSON 文件
    json_file_path = os.path.join(log_directory, json_file_name)
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # 提取事件列表并去掉开始窗口的解析
    trace_events = data.get('traceEvents', [])[1:]  # 去除第一条
    
    # 定义需要的类别和对应的CSV文件名
    categories = {
        "cuda_runtime": "cuda_runtime.csv",
        "gpu_memcpy": "gpu_memcpy.csv",
        "gpu_memset": "gpu_memset.csv",
        "cuda_driver": "cuda_driver.csv",
        "kernel": "kernel.csv",
        "conc_kernel": "conc_kernel.csv"
    }
    
    # 为每个 JSON 文件创建一个专属目录，名称可能是 JSON 文件名去掉 '.json'
    output_sub_dir = os.path.join(output_directory, os.path.splitext(json_file_name)[0])
    if not os.path.exists(output_sub_dir):
        os.makedirs(output_sub_dir)
    
    # 准备存储每个类别的数据
    categorized_events = defaultdict(list)
    
    # 根据类别分类数据
    for event in trace_events:
        cat = event.get('cat')
        if cat in categories:
            categorized_events[cat].append(event)
    
    # 对每个分类，写入到相应的CSV文件
    for cat in categories.keys():  # 遍历每一个可能的分类
        events = categorized_events[cat]  # 获取该类别的事件（可能为空）
        csv_file_name = os.path.join(output_sub_dir, categories[cat])
        
        # 提取所有可能的字段名，确保空文件也有头部
        header = set()
        for event in events:
            header.update(event.keys())
        
        # 将 header 转换为列表以确定顺序
        header = list(header)

        # 写入 CSV 文件，包含表头即使没有事件
        with open(csv_file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # 写入表头
            csv_writer.writerow(header)
            
            # 写入每个事件
            for event in events:
                row = [json.dumps(event.get(h, '')) if isinstance(event.get(h, ''), dict) else event.get(h, '') for h in header]
                csv_writer.writerow(row)
    
    print(f"文件 {json_file_name} 已成功生成在 {output_sub_dir} 目录下。")

print("所有文件转换完成。")
