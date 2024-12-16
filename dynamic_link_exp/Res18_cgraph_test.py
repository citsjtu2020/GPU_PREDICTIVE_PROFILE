import os.path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torchvision
# import async_cupti_trace
# import async_cupti_profile
import compute_graph_analysis
from compute_graph_analysis import qualified_tracer
import torch.fx
from compute_graph_analysis.prop_interpreter import ShapeProp

from compute_graph_analysis.utils import traverse_submodules

# 定义一个函数来遍历模块
# def traverse_submodules(module):
#     # print(module)  # 这里可以放置你想要对模块执行的操作
#     print(len(list(module.named_children())))
#     for submodule in module.named_children():
#         print((submodule))
#         print(type(submodule[1]))
#         traverse_submodules(submodule[1])



def Conv3x3BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class VGG(nn.Module):
    def __init__(self, block_nums, num_classes=1000):
        super(VGG, self).__init__()

        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        self._init_params()

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels, out_channels))
        for i in range(1, block_num):
            layers.append(Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
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
        x = x.view(x.size(0), -1)
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

    model = torchvision.models.resnet18()

    # model.to(device="cuda:0")

    input = torch.randn(5, 3, 224, 224)
    # input.to(device="cuda:0")
    # input = input.cuda()

    # print(model)

    do_module_qual = True

    do_shape_info = True

    out_path = "c_graph_result/resnet18"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    aim_metas = "shape,dtype,requires_grad,stride,memory_format,is_quantized,qparams"


    # pio_meta_list = aim_metas.strip().split(",")

    qual_tracer = qualified_tracer.QualifiedTracer()

    traced_model = qual_tracer.trace(model)

    qual_tracer2 = qualified_tracer.QualifiedTracer()

    # print(type(model.named_children()))

    # tmp_children = list(list(list(model.named_children())[1][1].named_children())[1][1].named_children())[1]
    # print(tmp_children[1].__str__())
    # traced_model2 = qual_tracer.trace((tmp_children[1]))
    # print(traced_model2)


    t1 = traverse_submodules(model,"")
    print(t1)
    print(len(t1))

    traver_model_dict = traverse_submodules(model)

    out_res_pdf2 = qual_tracer.static_graph_analysis(traced_model, out_qualified=do_module_qual,traver_m_dict=traver_model_dict)

    out_res_pdf2.to_csv(os.path.join(out_path,"res_static_graph.csv"),index=False)
    print(out_res_pdf2.shape)
    print(out_res_pdf2.columns)



    if do_shape_info:
        pio_meta_list = aim_metas.strip().split(",")
        interp = ShapeProp(torch.fx.GraphModule(model, traced_model))
        aim_meta_list = interp.find_aim_meta_list(pio_meta_list=pio_meta_list)
        print(aim_meta_list)

        interp.propagate(input)

        tensor_res = interp.catch_graph_tensor_prop(aim_meta_list=aim_meta_list)
        tensor_res.to_csv(os.path.join(out_path,"res_tensor_graph.csv"),index=False)

        print(tensor_res.shape)
        print(tensor_res.columns)




    # torchvision.models.vgg16_bn()
    # InitialCallbackProfiler(int numRanges=10,int kernelPerRange=1,int numPassPerSess=-1,string outputPath="profile_results",
    # string inputMetric="smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",int autoRange=1,int kernelReplay=1)
    # async_cupti_trace.InitializeTrace()
    # async_cupti_profile.InitialCallbackProfiler(-1,1,-1,"sampleCUPTI_profile",
    #                                             "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",1,1)



    # out = model(input)
    # # async_cupti_profile.FinalizeCallbackProfiler()
    # async_cupti_trace.FiniTrace("trace_no_profiler_res")
    # print(out.shape)