from typing import Callable

import torch


def parse_if_node(model):
    for node in model.graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.name == "else_branch":
                    model.graph.output.extend(attr.g.output)
                    length = len(attr.g.node)
                    for j, n in enumerate(attr.g.node):
                        if j == length-1:
                            n.output[0] = node.output[0]
                            
                        model.graph.node.extend([n])
            model.graph.node.remove(node)
    return model

def delete_node(node_names, model):
    for node in model.graph.node:
        if node.name in node_names:
            model.graph.node.remove(node)
    return model


def collate_fn(device) -> Callable:
    def inner(batch):
        sample = []
        for b in batch:
            sample.append(b[0])
        sample = torch.stack(sample).to(device)
        return sample

    return inner
