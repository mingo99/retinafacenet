from collections import OrderedDict

import numpy as np
import onnx
import onnxruntime as rt
import torch
from onnx import load_model, save_model

# 该设置可以将每层tensor完整地输出，而非输出部分（即省略中间，只显示收尾）
# tips：终端中通常不能全部显示大尺寸的tensor，可以重定向到文本中~
np.set_printoptions(threshold=np.inf)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def read_file(N, C, H, W, bin_path):
    input_size = N * C * H * W
    file = open(bin_path, "rb")
    data = np.fromfile(file, dtype=np.uint8)
    data = data.astype(np.float32) / 255
    data_tensor = torch.from_numpy(data[0:input_size])
    return data_tensor.reshape(N, C, H, W)


# 加载模型
model = onnx.load("./model.onnx")

for node in model.graph.node:
    for output in node.output:
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        print("node is: ", onnx.ValueInfoProto(name=output))

# 将指定结点打印出来，检查是否是自己想要的
print(model.graph.output)

# 将模型进行序列化为二进制格式
session = rt.InferenceSession(model.SerializeToString())

# 读取模型的输入
x1 = read_file(1, 1, 128, 128, "./input1.bin")
# x2 = read_file(1, 1, 224, 224, "./input2.bin")

# single input
ort_inputs = {session.get_inputs()[0].name: to_numpy(x1)}
# multiple inputs
# ort_inputs = {session.get_inputs()[0].name: to_numpy(x1), session.get_inputs()[1].name: to_numpy(x2)}
# 执行前向推理，保存结果
ort_out = session.run(None, ort_inputs)

print(ort_out)

# 获取前面循环中指定的所有节点输出
outputs = [x.name for x in session.get_outputs()]
# 将输出压缩在字典中，这样便于通过中间层名字作为key，获取具体的输出tensor值
ort_out = OrderedDict(zip(outputs, ort_out))

# print(len(ort_out))
# print(ort_out)
# 通过中中间层名称访问其输出tensor
# print(ort_out["layer_conv3"])
