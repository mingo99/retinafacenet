from typing import List

import torch
from ppq.executor.op.torch.base import TorchBackendContext, ASSERT_NUM_OF_INPUT, VALUE_TO_EXECUTING_DEVICE
from ppq.executor.torch import OPERATION_FORWARD_TABLE
from ppq.IR import Operation

def If_forward(
    op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs
) -> torch.Tensor:
    condition = values[0]
    then_branch = op.attributes.get("then_branch",[])
    else_branch = op.attributes.get("else_branch",[])
    print(f"condition: {condition}")
    # if condition:
    #     return then_branch.output[0]
        # for node in then_branch.node:
        #     values = OPERATION_FORWARD_TABLE[op.platform][node.op_type](node)
        #     print(values)
    # else:
    #     return else_branch.output[0]
        # for node in else_branch.node:
        #     values = OPERATION_FORWARD_TABLE[op.platform][node.op_type](node, values, ctx, **kwargs)
    # print(f"values: {values}")
    return values[0]

def ReduceMin_forward(
    op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs
) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get("axes", None)
    keepdim = bool(op.attributes.get("keepdims", 1))
    if len(input_value) == 0:
        output = input_value
    else:
        if dim is None:
            #  The default is to reduce over all the dimensions of the input tensor
            output = torch.min(input_value)
            if keepdim:
                output = output.reshape([1] * input_value.dim())
        else:
            output, _ = torch.min(input_value, dim=dim[0], keepdim=keepdim)
    return output

def ReduceProd_forward(
    op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs
) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get("axes", None)
    keepdim = bool(op.attributes.get("keepdims", 1))
    if len(input_value) == 0:
        output = input_value
    else:
        if dim is None:
            #  The default is to reduce over all the dimensions of the input tensor
            output = torch.prod(input_value)
            if keepdim:
                output = output.reshape([1] * input_value.dim())
        else:
            output, _ = torch.prod(input_value, dim=dim[0], keepdim=keepdim)
    return output

def Ceil_forward(
    op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs
):
    input_data = values[0]
    output = torch.ceil(input_data)
    return output


def Xor_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    a, b = values
    return a ^ b
