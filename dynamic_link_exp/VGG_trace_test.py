import torch
import torch.nn as nn
import torchvision
import async_cupti_trace
import async_cupti_profile

import async_cupti_trace_for_pro


def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class VGG(nn.Module):
    def __init__(self, block_nums,num_classes=1000):
        super(VGG, self).__init__()

        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        self._init_params()

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Conv3x3BNReLU(out_channels,out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out

def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGG(block_nums)
    return model

def VGG19():
    block_nums = [2, 2, 4, 4, 4]
    model = VGG(block_nums)
    return model

if __name__ == '__main__':
    
    model = VGG16()
    
    model.to(device="cuda:0")
    print(model)
    torchvision.models.vgg16_bn()
    # InitialCallbackProfiler(int numRanges=10,int kernelPerRange=1,int numPassPerSess=-1,string outputPath="profile_results",
    # string inputMetric="smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",int autoRange=1,int kernelReplay=1)
    #async_cupti_trace.InitializeTrace()
    # out_path = async_cupti_profile.InitialCallbackProfiler(10,
    #                                             1,1,
    #                                             "sampleCUPTI_profile_ex_new",
    #                                             "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
    #                                                        1,1,0)
    # smsp__inst_executed_pipe_tensor_op_hmma,smsp__inst_executed_pipe_tensor_op_dmma,smsp__inst_executed_pipe_tensor_op_imma
    out_path = async_cupti_profile.InitialCallbackProfiler(10,
                                                1,1,
                                                "sampleCUPTI_profile_ex_sm_smsp_add",
                                                "smsp__pipe_fma_cycles_active.sum,smsp__pipe_fma_cycles_active.avg,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__pipe_fp64_cycles_active.sum,smsp__pipe_fp64_cycles_active.avg,smsp__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.avg,sm__pipe_tensor_op_hmma_cycles_active.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_imma_cycles_active.avg,sm__pipe_tensor_op_imma_cycles_active.sum,sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
                                                # "smsp__inst_executed.avg.per_cycle_active,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fp64.sum,smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg,sm__cycles_active.avg,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.sum,sm__pipe_tensor_cycles_active.avg,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
                                                # "smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active,",
                                                #"sm__cycles_elapsed.avg,sm__cycles_active.avg,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__pipe_tensor_cycles_active.sum,smsp__pipe_tensor_cycles_active.avg",
                                                # sm__pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active
                                                # "sm__inst_executed_pipe_tensor_op_dmma.sum,sm__inst_executed_pipe_tensor_op.sum,sm__inst_executed_pipe_tensor_op.avg.pct_of_peak_sustained_active",
                                                           1,1,0)
    # smsp__inst_executed.avg.per_cycle_active,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_fp64.sum,smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
    # sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.sum


    input = torch.randn(1,3,224,224)
    input.to(device="cuda:0")
    input = input.cuda()
    out = model(input)
    try:
        async_cupti_profile.FinalizeCallbackProfiler()
    except Exception as ee:
        print(ee)

    async_cupti_trace_for_pro.InitializeTrace()

    # input = torch.randn(1, 3, 224, 224)
    # input.to(device="cuda:0")
    # input = input.cuda()
    out = model(input)
    try:
        async_cupti_trace_for_pro.FiniTrace(out_path)
    except Exception as ee:
        print(ee)


    #async_cupti_trace.FiniTrace("trace_no_profiler_res2")
    print(out.shape)
