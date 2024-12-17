import os
import json
import csv
from collections import defaultdict

def convert_json_to_csv(output_directory, log_directory):
    # 检查并创建主目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 找出 log 目录下所有 JSON 文件
    json_files = [f for f in os.listdir(log_directory) if f.endswith('.json')]

    # 遍历每个 JSON 文件
    for json_file_name in json_files:
        print(f"正在处理文件 {json_file_name}...")
        # 读取 JSON 文件
        json_file_path = os.path.join(log_directory, json_file_name)
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # 提取 traceEvents 列表
        trace_events = data.get('traceEvents', [])
        
        # 为每个 JSON 文件创建一个专属目录，名称可能是 JSON 文件名去掉 '.json'
        output_sub_dir = os.path.join(output_directory, os.path.splitext(json_file_name)[0])
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        
        # 准备存储每个类别的数据
        categorized_events = defaultdict(list)
        
        # 根据类别分类数据，并展开args字段
        for event in trace_events:
            cat = event.get('cat')
            if cat:
                # 展开args字段
                if 'args' in event:
                    for key, value in event['args'].items():
                        event[key] = value
                    del event['args']
                
                categorized_events[cat].append(event)
        
        # 对每个分类，写入到相应的CSV文件
        for cat, events in categorized_events.items():  # 遍历每一个可能的分类
            csv_file_name = os.path.join(output_sub_dir, f"{cat}.csv")
            
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
                    row = [json.dumps(event.get(h, '')) if isinstance(event.get(h, ''), (dict, list)) else event.get(h, '') for h in header]
                    csv_writer.writerow(row)
        
        print(f"文件 {json_file_name} 已成功转化为 {output_sub_dir} 目录下的csv")

    print("所有文件转换完成。")
