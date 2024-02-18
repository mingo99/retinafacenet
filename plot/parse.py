import re


def parse_lr(file_path: str):
    pattern = re.compile(r"lr: (\d+\.\d+)")
    lrs = []
    with open(file_path, "r") as f:
        log = f.read()
    
    matches = pattern.findall(log)
    for match in matches:
        lrs.append(float(match))
    return lrs

def parse_loss(file_path: str):
    total_pattern = re.compile(r"loss: (\d+\.\d+) \((\d+\.\d+)\)")
    bbox_pattern = re.compile(r"bbox_regression: (\d+\.\d+) \((\d+\.\d+)\)")
    cls_pattern = re.compile(r"classification: (\d+\.\d+) \((\d+\.\d+)\)")
    keyp_pattern = re.compile(r"keyp_regression: (\d+\.\d+) \((\d+\.\d+)\)")
    pattern = [total_pattern, bbox_pattern, cls_pattern, keyp_pattern]

    losses = []

    with open(file_path, "r") as f:
        log = f.read()

    for p in pattern:
        matches = p.findall(log)
        loss_list = []
        for match in matches:
            (
            _,
            avg_loss,
            ) = map(float, match)
            loss_list.append(avg_loss)
        losses.append(loss_list)

    return losses
