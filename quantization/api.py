from typing import Any, Callable, List

import onnx
import torch
from ppq.api import dump_torch_to_onnx, quantize_onnx_model
from ppq.api.setting import QuantizationSetting
from ppq.core import TargetPlatform, empty_ppq_cache
from ppq.IR import BaseGraph
from torch.utils.data import DataLoader

from datasets.coco import get_dataset, get_transform

from .util import collate_fn, delete_node, parse_if_node


def get_calibration_dataloader(args):
    print("Loading calibration data...")
    dataset, _ = get_dataset(
        "coco",
        "calibration",
        2023,
        get_transform(False, args),
        args.data_path,
        "keypoints",
    )
    data_loader= torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn(args.device),
    )

    return data_loader

def model_trimming(onnx_import_file,onnx_export_file):
    model = onnx.load(onnx_import_file)
    del_names = ["Shape_2537","ReduceProd_2538","Equal_2540","Cast_2541","Shape_3844","ReduceProd_3845","Equal_3847","Cast_3848"]
    model = parse_if_node(model)
    model = delete_node(del_names, model)
    del model.graph.output[-2:]
    onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_export_file)

@ empty_ppq_cache
def quantize_torch_model(
    model: torch.nn.Module,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    platform: TargetPlatform,
    input_dtype: torch.dtype = torch.float,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    inputs: List[Any] = None,
    do_quantize: bool = True,
    onnx_export_file: str = 'onnx.model',
    device: str = 'cuda',
    verbose: int = 0,
    ) -> BaseGraph:
    """量化一个 Pytorch 原生的模型 输入一个 torch.nn.Module 返回一个量化后的 PPQ.IR.BaseGraph.

        quantize a pytorch model, input pytorch model and return quantized ppq IR graph
    Args:
        model (torch.nn.Module): 被量化的 torch 模型(torch.nn.Module) the pytorch model

        calib_dataloader (DataLoader): 校准数据集 calibration dataloader

        calib_steps (int): 校准步数 calibration steps

        collate_fn (Callable): 校准数据的预处理函数 batch collate func for preprocessing

        input_shape (List[int]): 模型输入尺寸，用于执行 jit.trace，对于动态尺寸的模型，输入一个模型可接受的尺寸即可。
            如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                a list of ints indicating size of input, for multiple inputs, please use
                                keyword arg inputs for direct parameter passing and this should be set to None

        input_dtype (torch.dtype): 模型输入数据类型，如果模型存在多个输入，则需要使用 inputs 变量进行传参，此项设置为 None
                                the torch datatype of input, for multiple inputs, please use keyword arg inputs
                                for direct parameter passing and this should be set to None

        setting (OptimSetting): 量化配置信息，用于配置量化的各项参数，设置为 None 时加载默认参数。
                                Quantization setting, default setting will be used when set None

        inputs (List[Any], optional): 对于存在多个输入的模型，在Inputs中直接指定一个输入List，从而完成模型的tracing。
                                for multiple inputs, please give the specified inputs directly in the form of
                                a list of arrays

        do_quantize (Bool, optional): 是否执行量化 whether to quantize the model, defaults to True, defaults to True.

        platform (TargetPlatform, optional): 量化的目标平台 target backend platform, defaults to TargetPlatform.DSP_INT8.

        device (str, optional): 量化过程的执行设备 execution device, defaults to 'cuda'.

        verbose (int, optional): 是否打印详细信息 whether to print details, defaults to 0.

    Raises:
        ValueError: 给定平台不可量化 the given platform doesn't support quantization
        KeyError: 给定平台不被支持 the given platform is not supported yet

    Returns:
        BaseGraph: 量化后的IR，包含了后端量化所需的全部信息
                   The quantized IR, containing all information needed for backend execution
    """
    # dump pytorch model to onnx
    dump_torch_to_onnx(model=model, onnx_export_file=onnx_export_file,
        input_shape=input_shape, input_dtype=input_dtype,
        inputs=inputs, device=device)

    model_trimming(onnx_export_file,onnx_export_file)

    return quantize_onnx_model(onnx_import_file=onnx_export_file,
        calib_dataloader=calib_dataloader, calib_steps=calib_steps, collate_fn=collate_fn,
        input_shape=input_shape, input_dtype=input_dtype, inputs=inputs, setting=setting,
        platform=platform, device=device, verbose=verbose, do_quantize=do_quantize)
