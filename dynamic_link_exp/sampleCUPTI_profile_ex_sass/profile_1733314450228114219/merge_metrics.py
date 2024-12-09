import csv
import os
import re

# 定义基础目录和输出文件
base_dir = 'dev_0_ctx_1'
output_file = 'merged_metric_results.csv'

# 定义列名
header = ['session_id', 'range_id', 'launchId', 'kernel', 'metric_name', 'value']

# 自然排序函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# 打开输出文件并写入标题行
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # 获取并排序 dev_0_ctx_1 文件夹下的所有 sess_* 子文件夹
    subdirs = sorted([d for d in os.listdir(base_dir) if d.startswith('sess_')], key=natural_sort_key)

    # 遍历排序后的子文件夹
    for subdir in subdirs:
        metric_file_path = os.path.join(base_dir, subdir, 'metric_results.csv')
        if os.path.exists(metric_file_path):
            print(f"Processing {metric_file_path}")
            with open(metric_file_path, 'r') as mf:
                reader = csv.reader(mf)
                next(reader)  # 跳过标题行
                for row in reader:
                    session_id = row[0]  # 使用子文件夹名称作为 session_id
                    range_id = row[1]
                    launch_id = row[2]
                    kernel = row[3]
                    metric_name = row[4]
                    value = row[5]
                    writer.writerow([session_id, range_id, launch_id, kernel, metric_name, value])

print(f"Merged metric results have been written to {output_file}")
