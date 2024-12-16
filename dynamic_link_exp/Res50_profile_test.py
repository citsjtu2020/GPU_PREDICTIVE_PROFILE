import torch
import torchvision.models
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# from d2l import torch as d2l
import async_cupti_trace
import async_cupti_profile

import async_cupti_trace_for_pro
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 下载并配置数据集
trans = transforms.Compose(
    [transforms.Resize((96, 96)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(
    root=r'/data/dataset_example/cifar-10', train=True, transform=trans, download=True)
test_dataset = datasets.CIFAR10(
    root=r'/data/dataset_example/cifar-10', train=False, transform=trans, download=True)

total_metric_list = ['smsp__pipe_fp64_cycles_active.sum',
                     'sm__pipe_tensor_cycles_active.avg',
                     'smsp__sass_thread_inst_executed_op_fp32_pred_on.sum',
                     'smsp__inst_executed_pipe_fp64.sum',
                     'sm__pipe_tensor_cycles_active.sum',
                     'smsp__inst_executed.sum',
                     'smsp__sass_thread_inst_executed_op_hmul_pred_on.sum',
                     'smsp__sass_thread_inst_executed_op_fp64_pred_on.sum',
                     'sm__pipe_tensor_op_hmma_cycles_active.avg',
                     'smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active',
                     'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
                     'sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active',
                     'sm__cycles_elapsed.avg',
                     'sm__cycles_active.avg',
                     'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
                     'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
                     'smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active',
                     'smsp__sass_thread_inst_executed_op_hfma_pred_on.sum',
                     'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active',
                     'smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active',
                     'sm__pipe_tensor_op_hmma_cycles_active.sum',
                     'smsp__inst_executed_pipe_fma.sum',
                     'smsp__pipe_fma_cycles_active.sum',
                     'smsp__inst_executed.avg.per_cycle_active',
                     'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum',
                     'smsp__sass_thread_inst_executed_op_hadd_pred_on.sum',
                     'sm__pipe_tensor_op_imma_cycles_active.sum',
                     'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum',
                     'smsp__pipe_fp64_cycles_active.avg',
                     'sm__cycles_active.avg.pct_of_peak_sustained_elapsed',
                     'smsp__pipe_fma_cycles_active.avg',
                     'sm__pipe_tensor_op_imma_cycles_active.avg',
                     'smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
                     'smsp__sass_thread_inst_executed_op_fp16_pred_on.sum',
                     'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
                     'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum']


def train(net, train_iter, test_iter, epochs, lr, device, upp_limit_iter=10, profile_iter=1,
          out_path="Resnet50_trace", metrics="smsp__pipe_fma_cycles_active.sum",
          numRanges=10, kernelPerRange=1, numPassPerSess=1, autoRange=1, kernelReplay=1, base_start=0):
    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         nn.init.xavier_uniform_(m.weight)
    # net.apply(init_weights)
    print(f'Training on:[{device}]')
    net.to(device)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    now_iter = 0
    profile_base_path = out_path

    for epoch in range(epochs):
        # 训练损失之和，训练准确率之和，样本数
        # metric = d2l.Accumulator(3)

        net.train()
        for i, (X, y) in enumerate(train_iter):
            # timer.start()
            print(i)
            if i == profile_iter:
                # InitialCallbackProfiler(int numRanges=10,int kernelPerRange=1,int numPassPerSess=-1,string outputPath="profile_results",
                # string inputMetric="smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
                # int autoRange=1,int kernelReplay=1,uint64_t base_start=0)
                profile_base_path = async_cupti_profile.InitialCallbackProfiler(numRanges,
                                                                                kernelPerRange, numPassPerSess,
                                                                                out_path,
                                                                                metrics,
                                                                                autoRange, kernelReplay, base_start)

            if i == (profile_iter + 1):
                async_cupti_trace_for_pro.InitializeTrace()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            if i == profile_iter:
                try:
                    async_cupti_profile.FinalizeCallbackProfiler()
                except Exception as ee:
                    print(ee)

            if i == (profile_iter + 1):
                try:
                    async_cupti_trace_for_pro.FiniTrace(profile_base_path)
                except Exception as ee:
                    print(ee)

            now_iter += 1

            if now_iter >= upp_limit_iter:
                break


# 配置数据加载器
batch_size = 10
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet50()
epochs, lr = 20, 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

defaultMetric = 'smsp__inst_executed.sum'

pro_aim = "sass"  # sm_ # smsp

inputMetric = ""

use_metric = []

for jj in total_metric_list:
    if pro_aim.lower() in jj.lower():
        if "sass" not in pro_aim.lower():
            if "sass" not in jj.lower():
                use_metric.append(jj)
        else:
            use_metric.append(jj)

for h in range(len(use_metric)):
    if h == 0:
        inputMetric = use_metric[h]
        if len(use_metric) > 1:
            inputMetric = inputMetric + ","
    else:
        if h < len(use_metric) - 1:
            inputMetric = inputMetric + use_metric[h]
            inputMetric = inputMetric + ","
        else:
            inputMetric = inputMetric + use_metric[h]

if not inputMetric:
    inputMetric = defaultMetric

out_path = "Profile_result"
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_path = "Profile_result/Resnet50"
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_path = os.path.join(out_path, f"out_result_{pro_aim.lower()}")

print(f"Aim Metric are: {inputMetric}")
print(f"Output Path: {out_path}")

train(model, train_loader, test_loader, epochs, lr, device, upp_limit_iter=10, profile_iter=1,
      out_path=out_path, metrics=inputMetric,
      numRanges=10, kernelPerRange=1, numPassPerSess=1, autoRange=1, kernelReplay=1, base_start=0)
