# profiler_utils.py

import torch
import os
from convert_json_to_csv import convert_json_to_csv

def py_tracing_forward(model, input_data, model_name, wait=1, warmup=0, active=1, repeat=1):
    total = (wait + warmup + active) * repeat

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=wait,  # 等待周期
            warmup=warmup,  # 预热周期
            active=active,  # 活动周期
            repeat=repeat   # 重复次数
        ),
        record_shapes=True,  # 记录输入形状
        with_stack=True  # 记录堆栈跟踪
    ) as prof:
        for _ in range(total):  
            output = model(input_data)
            # loss = output.sum()  # 假设损失是输出的和
            # loss.backward()  # 反向传播
            prof.step()

    # 指定路径和名字保存 JSON
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    json_path = os.path.join(log_dir, f'{model_name}_forward.json')
    prof.export_chrome_trace(json_path)

    # JSON to CSV
    log_directory = 'log'
    output_directory = 'log_csv'
    convert_json_to_csv(output_directory, log_directory)

    print(f"Profiling completed. Trace log saved to '{json_path}'")
    print(f"CSV files saved to '{output_directory}'")


def py_tracing_backward(model, input_data, model_name, wait=0, warmup=0, active=1, repeat=1):
    total = (wait + warmup + active) * repeat
    output = model(input_data)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=wait,  # 等待周期
            warmup=warmup,  # 预热周期
            active=active,  # 活动周期
            repeat=repeat   # 重复次数
        ),
        record_shapes=True,  # 记录输入形状
        with_stack=True  # 记录堆栈跟踪
    ) as prof:
        for _ in range(total):  
            loss = output.sum() 
            loss.backward()  
            prof.step()

    # 指定路径和名字保存 JSON
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    json_path = os.path.join(log_dir, f'{model_name}_backward.json')
    prof.export_chrome_trace(json_path)

    # JSON to CSV
    log_directory = 'log'
    output_directory = 'log_csv'
    convert_json_to_csv(output_directory, log_directory)

    print(f"Profiling completed. Trace log saved to '{json_path}'")
    print(f"CSV files saved to '{output_directory}'")