import torch
import torchvision
import torch.nn as nn
from typing import OrderedDict, List

import models


def fuse_conv_bn(): ...

def export_onnx():
    model = models.get_model("retinafacenet_resnet50_fpn").eval().cuda()

    dummy_input = torch.rand((1, 3, 800, 800), device="cuda")

    torch.onnx.export(
        model,
        dummy_input,
        "quantization/model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13
    )

if __name__ == "__main__":
    export_onnx()
