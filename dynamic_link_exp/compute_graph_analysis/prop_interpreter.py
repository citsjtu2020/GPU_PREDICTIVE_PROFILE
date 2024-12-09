import traceback
from typing import Any, Dict, NamedTuple, Optional, Tuple

import pandas as pd
import torch
import torch.fx
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx._compatibility import compatibility
from torch.fx.node import map_aggregate, Node
from compute_graph_analysis import utils
__all__ = ["TensorMetadata", "ShapeProp"]








@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape: torch.Size = None
    dtype: torch.dtype = None
    requires_grad: bool = False
    stride: Tuple[int, ...] = None
    memory_format: Optional[torch.memory_format] = None

    # Quantization metadata
    is_quantized: bool = False
    qparams: Dict[str, Any] = {}


def _extract_tensor_metadata(
        result: torch.Tensor,
        include_contiguity=True) -> TensorMetadata:
    """
        Extract a TensorMetadata NamedTuple describing `result`.
        """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride() if not is_sparse_any(result) else ()

    memory_format = None

    if include_contiguity and not is_sparse_any(result):
        '''
        在 PyTorch 中，torch.contiguous_format, torch.channels_last, 和 torch.channels_last_3d 是与张量内存布局相关的概念。这些格式定义了多维张量（尤其是图像数据）在内存中的存储方式，这对于优化某些特定操作（如卷积运算）的性能非常重要。下面是对这三种内存格式的解释：

        torch.contiguous_format:
        这是默认的内存布局格式。在这种格式下，张量的元素在内存中是连续存储的，并且遵循C语言风格的顺序（行优先）。这意味着对于一个形状为 (N, C, H, W) 的4D张量，其内存布局是按照 N、C、H、W 的顺序依次排列。
        当使用这种格式时，张量通常是“连续”的，即每个维度上的步幅（stride）反映了张量的实际大小。例如，对于一个形状为 (2, 3, 4, 5) 的张量，其步幅可能是 (60, 20, 5, 1)。
        torch.channels_last:
        这种格式主要用于二维图像数据（通常是形状为 (N, C, H, W) 的张量），其中通道（C）维度被移动到最后。因此，内存布局变成了 (N, H, W, C)。
        使用这种格式可以加速某些类型的卷积操作，因为现代GPU架构通常对这种布局进行了优化。例如，在执行卷积时，它可以减少内存访问模式的跳跃性，从而提高缓存效率和计算速度。
        要将张量转换为此格式，可以使用 .to(memory_format=torch.channels_last) 方法。
        torch.channels_last_3d:
        类似于 torch.channels_last，但适用于三维图像数据（通常是形状为 (N, C, D, H, W) 的张量），其中通道（C）维度也被移动到最后。因此，内存布局变为 (N, D, H, W, C)。
        这种布局同样是为了优化三维卷积等操作，特别是在处理视频或体积数据时。通过调整内存布局，可以更好地利用GPU的硬件特性来加速计算。
        同样地，要将张量转换为此格式，可以使用 .to(memory_format=torch.channels_last_3d) 方法。
        '''
        memory_formats = {
            torch.contiguous_format,
            torch.channels_last,
            torch.channels_last_3d,
        }
        for query_format in memory_formats:
            if result.is_contiguous(memory_format=query_format):
                memory_format = query_format
                break

    is_quantized = result.is_quantized

    qparams: Dict[str, Any] = {}

    if is_quantized:
        qscheme = result.qscheme()
        '''
        torch.per_tensor_affine 和 torch.per_tensor_symmetric 是 PyTorch 中与量化相关的概念，它们是两种不同的量化方案，用于在模型量化过程中定义如何对张量进行量化。这两种方法都属于逐张量（per-tensor）量化策略，意味着整个张量使用相同的量化参数（如缩放因子和零点）。下面是这两个术语的简要解释：

        torch.per_tensor_affine:
        这种量化方式使用仿射变换来将浮点值映射到整数表示。它适用于那些分布不对称的数据。
        量化公式可以表示为：
        𝑄=round(𝑋/𝑠𝑐𝑎𝑙𝑒+𝑧𝑒𝑟𝑜_𝑝𝑜𝑖𝑛𝑡)
        Q=round(X/scale+zero_point)，其中 𝑋 是原始浮点数值，
        scale 是缩放因子，
        zero_point 是一个整数偏移量，用来保证量化后的值能够准确地表示0附近的值。
        该方法通常用于激活函数的输出或权重等需要保持正负值的情况。

        torch.per_tensor_symmetric:
        对于数据分布相对对称的情况，可以采用对称量化方法。这种情况下，零点通常是0，这意味着量化过程是对原点对称的。
        量化公式简化为：𝑄=round(𝑋/𝑠𝑐𝑎𝑙𝑒)
        Q=round(X/scale)，这里没有显式的零点偏移。
        由于对称性的特点，这种方法通常用于权重的量化，尤其是当权重的分布接近对称时，可以减少量化带来的误差。
        在PyTorch中，这些量化模式通常作为量化配置的一部分被指定，例如在创建量化观测器（Quantization Observer）时。观测器会根据给定的模式计算出合适的量化参数，并应用于后续的量化操作。
        '''
        qparams["qscheme"] = qscheme


        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in {
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
            torch.per_channel_symmetric,
        }:
            '''
            在 PyTorch 中，torch.per_channel_affine, torch.per_channel_affine_float_qparams, \
            和 torch.per_channel_symmetric 是与量化相关的参数配置方案。
            这些配置定义了如何对张量进行逐通道（per-channel）量化，特别是针对卷积层权重等场景。下面是对这三种量化方案的解释：

            1.torch.per_channel_affine:
            这种量化方式使用逐通道仿射变换来将浮点值映射到整数表示。每个通道都有自己的缩放因子（scale）和零点（zero_point），
            这意味着不同通道可以根据其数据分布独立地进行量化。
            量化公式为：
            𝑄=round(𝑋/𝑠𝑐𝑎𝑙𝑒+𝑧𝑒𝑟𝑜_𝑝𝑜𝑖𝑛𝑡)
            Q=round(X/scale+zero_point)，其中 𝑋是原始浮点数值，
            scale 和 zero_point 都是针对每个通道单独确定的。
            这种方法通常用于卷积层的权重，因为不同输出通道的数据分布可能非常不同，采用逐通道量化可以更好地保留信息。

            2.torch.per_channel_affine_float_qparams:
            这种量化方式类似于 torch.per_channel_affine，
            但允许缩放因子（scale）和零点（zero_point）以浮点数的形式存储。这对于需要高精度量化参数的情况特别有用。
            由于量化参数是浮点数，这种方法主要用于那些需要较高精度的情况，尤其是在某些特定硬件平台上。
            它同样适用于卷积层的权重，特别是在需要更精细控制量化参数的情况下。

            3. torch.per_channel_symmetric:
            这是一种逐通道对称量化的方法，假设每个通道的数据分布是对称的。这种情况下，零点通常是0，因此量化过程是对原点对称的。
            量化公式简化为：
            𝑄=round(𝑋/𝑠𝑐𝑎𝑙𝑒)
            Q=round(X/scale)，这里没有显式的零点偏移。
            对称量化特别适合于权重的量化，尤其是当权重的分布接近对称时，它可以有效地减少量化带来的误差。
            '''

            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream



            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams
    )


@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """

    def __init__(self, gm, fake_mode=None):
        super().__init__(gm)
        self.valid_metas = ["shape","dtype",
                            "requires_grad","stride",
                            "memory_format","is_quantized","qparams"]
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor

            # Note:
            # We need fake execution cause the inputs are fake, however, we cannot fakify the module
            # - because we need to write to the tensor_meta of the real module. So we fakify to
            # produce a result (L131 below), to extract tensor meta, and then keep going.
            #
            # If we were to fakify, we would write to the wrong node, and then downstream fusion
            # would be missing the tensor_meta.
            #
            # See torch/_inductor/overrides.py for where this is called upstream of fusion.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module

    def find_aim_meta_list(self,pio_meta_list):
        aim_meta_list = ["shape","dtype"]
        for pm in pio_meta_list:
            if pm:
                if pm.strip().lower() in self.valid_metas:
                    aim_meta_list.append(pm)

        aim_meta_list = list(set(aim_meta_list))

        return aim_meta_list[:]

    def run_node(self, n: Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with " f"meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta

        n.meta["type"] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
            fake_args = [
                self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in args
            ]
        else:
            fake_args = args
        return super().run(*fake_args)

    def catch_graph_tensor_prop(self, aim_meta_list=["shape", "dtype"]):
        total_out_res_pdf = pd.DataFrame()
        graph = self.graph

        for node in graph.nodes:
            node_res_pdf = utils.catch_node_tensor_prop(node, aim_meta_list=aim_meta_list)
            if total_out_res_pdf.shape[0] < 1:
                total_out_res_pdf = node_res_pdf.copy().reset_index(drop=True)
            else:
                total_out_res_pdf = pd.concat([total_out_res_pdf,
                                               node_res_pdf.reset_index(drop=True)], axis=0).reset_index(drop=True)
        return total_out_res_pdf


class ShapeProp2(torch.fx.Interpreter):
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph.result)









