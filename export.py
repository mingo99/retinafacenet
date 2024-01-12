from typing import List, OrderedDict

import torch
import torch.nn as nn
import torchvision

import models


def fuse_conv_bn():
    ...


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
        opset_version=13,
    )


def export_mobilefacenet():
    model = (
        models.get_model(
            "facenet_mobilev2", weights="./weights/facenet_mobilev2_state_dict.pth"
        )
        .eval()
        .cuda()
    )

    dummy_input = torch.rand((1, 3, 112, 112), device="cuda")

    torch.onnx.export(
        model,
        dummy_input,
        "quantization/mobilefacenet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13,
    )


if __name__ == "__main__":
    # export_onnx()
    export_mobilefacenet()
