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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 下载并配置数据集
trans = transforms.Compose(
    [transforms.Resize((96, 96)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(
    root=r'/data/dataset_example/cifar-10', train=True, transform=trans, download=True)
test_dataset = datasets.CIFAR10(
    root=r'/data/dataset_example/cifar-10', train=False, transform=trans, download=True)


def train(net, train_iter, test_iter, epochs, lr, device, upp_limit_iter=10, trace_iter=1, out_path="Resnet18_trace"):
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

    for epoch in range(epochs):
        # 训练损失之和，训练准确率之和，样本数
        # metric = d2l.Accumulator(3)

        net.train()
        for i, (X, y) in enumerate(train_iter):
            # timer.start()
            print(i)
            if i == trace_iter:
                async_cupti_trace.InitializeTrace()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            if i == trace_iter:
                try:
                    async_cupti_trace.FiniTrace(out_path)
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

model = torchvision.models.resnet101()
epochs, lr = 20, 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path = "Trace_result"

if not  os.path.exists(base_path):
    os.makedirs(base_path)

train(model, train_loader, test_loader, epochs, lr, device, upp_limit_iter=10, trace_iter=1, out_path="Trace_result/Resnet101")

#         test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
#         print(
#             f'Train Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%')
#     print(f'{metric[2] * epochs / timer.sum():.1f} examples/sec '
#           f'on: [{str(device)}]')
#     torch.save(net.state_dict(),
#                f"E:\\Deep Learning\\model\\ResNet-18_CIFAR-10_Epoch{epochs}_Accuracy{test_acc*100:.2f}%.pth")
#
# for i, (X, y) in enumerate(train_loader):
#     print(X.shape)
#     print(i)
#
#     break