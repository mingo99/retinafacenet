import numpy as np
import onnx
import onnxruntime
import torch
import torchinfo
import torchvision
from onnxsim import simplify
# from ppq.api import export
from torch.quantization import fuse_modules
from torchvision.models.resnet import Bottleneck

import models
# import quantization.api as util

fbnet = models.get_model("retinafacenet_resnet50_fpn").eval()
mbface = models.get_model("facenet_mobilev2").eval()
# retinanet = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).eval()
resnet50 = torchvision.models.resnet50(pretrained=True).eval()
# print(model.state_dict().keys())
#  bn_keys = []
#  for key in model.state_dict().keys():
#  if "bn" in key:
#  bn_keys.append(key)
#
#  print(bn_keys)
#
# torchinfo.summary(resnet50, (1, 3, 800, 800))
# torchinfo.summary(fbnet, (1, 3, 800, 800))
torchinfo.summary(mbface, (1, 3, 112, 112))
# torchinfo.summary(retinanet, (1, 3, 800, 800))
# ONNX_PATH = "./output/raw_onnx.model"
# ONNX_OUTPUT_PATH = "./output/model.onnx"
# onnx_model = onnx.load(ONNX_OUTPUT_PATH)
#
# # print(onnx_model.graph.output)
# for node in onnx_model.graph.node:
#     if node.name == "/transform/Sub_3":
#         print(node.attribute)
#         node.attribute[0].t.data_type = onnx.TensorProto.INT64
#
# onnx.save(onnx_model, "./output/modified_model.onnx")
