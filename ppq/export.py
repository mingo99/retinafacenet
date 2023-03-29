import torch
import torchvision
import torch.nn as nn
from typing import OrderedDict, List
import sys
# sys.path.append("..")
from model import retinafacenet_resnet50_fpn

class SimpRetinanet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2()

    def forward(self, images, targets=None):
        # transform the input
        images, targets = self.model.transform(images, targets)

        # get the features from the backbone
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.model.head(features)

        return head_outputs


def export_onnx():
    # model = SimpRetinanet().eval().cuda()
    model = retinafacenet_resnet50_fpn().eval().cuda()

    dummy_input = torch.rand((1, 3, 800, 800), device="cuda")

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13
    )

if __name__ == "__main__":
    export_onnx()
