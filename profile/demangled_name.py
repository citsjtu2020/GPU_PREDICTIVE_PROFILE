import cxxfilt
import json

def demangle_with_cxxfilt(symbol):
    try:
        return cxxfilt.demangle(symbol)
    except cxxfilt.InvalidName:
        return symbol

# 假设你有20个符号名存储在一个字典中，键是原始符号名，值是简短名称
symbols = {
    "_ZN14cask__5x_cudnn20computeOffsetsKernelILb0ELb0EEEvNS_20ComputeOffsetsParamsE": "computeOffsetsKernel",
    "_5x_cudnn_ampere_scudnn_128x64_relu_medium_nn_v1": "ampere_scudnn_128x64_relu",
    "_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastINS0_15CUDAFunctor_addIfEEEEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1_": "elementwise_kernel_gpu_kernel_impl_nocast",
    "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_21CUDAFunctorOnSelf_addIlEENS_6detail5ArrayIPcLi2EEEEEviT0_T1_": "vectorized_elementwise_kernel",
    "_ZN5cudnn25bn_fw_tr_1C11_kernel_NCHWIffiLi512ELb1ELi1ELb1EEEv17cudnnTensorStructPKT_S1_PS2_PKT0_S8_S6_S6_PS6_S9_S9_S9_S6_S6_": "bn_fw_tr_1C11_kernel_NCHW_cudnnTensorStruct",
    "_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_49_GLOBAL__N__d2ba64fb_16_TensorCompare_cu_71e06f4e19launch_clamp_scalarERNS_18TensorIteratorBaseEN3c106ScalarES6_NS0_6detail11ClampLimitsEENKUlvE_clEvENKUlvE5_clEvEUlfE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_": "vectorized_elementwise_kernel_TensorCompare_cu_launch_clamp_scalar",
    "_ZN5cudnn8winograd27generateWinogradTilesKernelILi0EffEEvNS0_27GenerateWinogradTilesParamsIT0_T1_EE": "generateWinogradTilesKernel",
    "_5x_cudnn_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1": "ampere_scudnn_winograd_128x128_relu",
    "_ZN2at6native52_GLOBAL__N__e57809e0_19_DilatedMaxPool2d_cu_6258b57421max_pool_forward_nchwIfEEviPKT_llliiiiiiiiiiPS3_Pl": "max_pool_forward_nchw",
    "_ZN5cudnn19engines_precompiled16nchwToNhwcKernelIfffLb0ELb1EL21cudnnKernelDataType_t2EEEvNS0_18nchw2nhwc_params_tIT1_EEPKT_PT0_": "engines_precompiled_nchwToNhwcKernel_Lb0ELb1E",
    "_ZN5cudnn24bn_fw_tr_1C11_singlereadIfLi512ELb1ELi1ELi2ELi0EEEvNS_18bn_fw_tr_1C11_argsIT_EE": "bn_fw_tr_1C11_singleread",
    "_ZN17cutlass__5x_cudnn6KernelI66cutlass_tensorop_s1688fprop_optimized_tf32_128x64_16x6_nhwc_align4EEvNT_6ParamsE": "Kernel_fprop_optimized16x6",
    "_ZN5cudnn19engines_precompiled16nhwcToNchwKernelIfffLb1ELb0EL21cudnnKernelDataType_t0EEEvNS0_18nhwc2nchw_params_tIT1_EEPKT_PT0_": "engines_precompiled_nhwcToNchwKernel_Lb1ELb0E",
    "_ZN17cutlass__5x_cudnn6KernelI66cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4EEvNT_6ParamsE": "Kernel_fprop_optimized32x3",
    "_ZN8internal5gemvx6kernelIiiffffLb0ELb1ELb1ELb0ELi7ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES5_S3_IfEfEEENSt9enable_ifIXntT5_EvE4typeET11_": "gemvx6kernel",
    "_ZN2at6native43_GLOBAL__N__50f1b6c4_10_Dropout_cu_0e96ed3824fused_dropout_kernel_vecIffjLi1ELi4EbEEvNS_4cuda6detail10TensorInfoIKT_T1_EENS5_IS6_S8_EENS5_IT4_S8_EES8_T0_NS_15PhiloxCudaStateE": "fused_dropout_kernel_vec",
    "_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi4ELi4ELb0ELb1E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_": "gemv2T_kernel_val",
    "sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize256x128x32_stage2_warpsize4x2x1_g1_tensor16x8x8_alignc4_execute_kernel__5x_cudnn": "sm86_xmma_fprop_implicit_gemm_tilesize256x128x32_stage2_warpsize4x2x1_g1_tensor16x8x8_execute_kernel",
    "sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage3_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel__5x_cudnn": "sm86_xmma_fprop_implicit_gemm_tilesize128x128x16_stage3_warpsize2x2x1_g1_tensor16x8x8_execute_kernel",
    "sm86_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x64x32_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_cudnn": "sm86_xmma_fprop_implicit_gemm_indexed_tilesize128x64x32_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel"
}

# 创建一个空字典来存储解码后的名称及其简短名称
demangled_symbols = {}

# 依次对每个符号名进行解码
for symbol, short_name in symbols.items():
    demangled_name = demangle_with_cxxfilt(symbol)
    demangled_symbols[symbol] = {
        "short_name": short_name,
        "demangled_name": demangled_name
    }

# 将结果保存到一个新的JSON文件中
output_file = "demangled_symbols.json"
with open(output_file, "w") as f:
    json.dump(demangled_symbols, f, indent=4)

print(f"Results saved to {output_file}")
