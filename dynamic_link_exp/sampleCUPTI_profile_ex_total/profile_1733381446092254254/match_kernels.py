import csv
import os

# 读取第一个文件的信息
kernel_info = {}
with open('kernel_launch_info.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    for row in reader:
        launch_id = int(row[1])
        kernel_name = row[0]
        kernel_info[launch_id] = kernel_name

# 定义函数来处理单个 metric_results.csv 文件
def process_metric_file(file_path):
    output_rows = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取标题行
        output_rows.append(header)
        for row in reader:
            if row[3] == '<unknown>':
                launch_id = int(row[2])
                if launch_id in kernel_info:
                    row[3] = kernel_info[launch_id]
                    print(f"Replaced <unknown> with {kernel_info[launch_id]} for launchId {launch_id} in {file_path}")
                else:
                    print(f"Could not find kernel name for launchId {launch_id} in {file_path}")
            output_rows.append(row)
    
    # 将结果写入新的文件
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
        print(f"Updated {file_path}")

# 遍历 dev_0_ctx_1 文件夹下的所有 sess_* 子文件夹
base_dir = 'dev_0_ctx_1'
for subdir in os.listdir(base_dir):
    if subdir.startswith('sess_'):
        metric_file_path = os.path.join(base_dir, subdir, 'metric_results.csv')
        if os.path.exists(metric_file_path):
            print(f"Processing {metric_file_path}")
            process_metric_file(metric_file_path)
            print(f"Processed {metric_file_path}")

print("All metric results files have been processed and unknown kernel names have been replaced.")
