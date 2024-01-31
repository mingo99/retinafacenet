import numpy as np
import onnx
import onnxruntime
import torch
import torchinfo
import torchvision
from onnxsim import simplify
from ppq.api import export
from torch.quantization import fuse_modules
from torchvision.models.resnet import Bottleneck

import models

# fbnet = models.get_model("retinafacenet_resnet50_fpn").eval()
# retinanet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).eval()
# print(model.state_dict().keys())
#  bn_keys = []
#  for key in model.state_dict().keys():
#  if "bn" in key:
#  bn_keys.append(key)
#
#  print(bn_keys)
#
# torchinfo.summary(fbnet, (1, 3, 800, 800))
# torchinfo.summary(retinanet, (1, 3, 800, 800))
ONNX_PATH = "./quantization/model.onnx"
onnx_model = onnx.load(ONNX_PATH)

for node in onnx_model.graph.node:
    if node.name == "/transform/Concat_7":
        onnx_model.graph.output.extend([onnx.ValueInfoProto(name=node.output[0])])
# print(onnx_model.graph.output)

try:
    import onnxruntime
except ImportError as e:
    raise Exception("Onnxruntime is not installed.")

sess = onnxruntime.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnxruntime_outputs = []
onnxruntime_outputs.append(
    sess.run(
        output_names=[name for name in onnx_model.graph.output],
        input_feed={"input.1": np.zeros([1, 3, 800, 800])},
    )
)
# print(onnxruntime_outputs)
outputs = sess.get_outputs()
print(outputs)
# if node.name == "/anchor_generator/ConstantOfShape":
#     print(node)

# fbnet = models.get_model(
#     "retinafacenet_resnet50_fpn", weights="./weights/retinafacenet_resnet50.pth"
# ).eval()
# fbnet.cuda()
# fbnet.fuse_model()
# dummy_input = torch.rand((1, 3, 800, 800), device="cuda")
#
# torch.onnx.export(
#     fbnet,
#     dummy_input,
#     "output/model.onnx",
#     input_names=["input.1"],
#     output_names=["output"],
#     opset_version=13,
# )
#
# print(fbnet)
