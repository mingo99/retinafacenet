from typing import Dict, List, Optional, Tuple

from torch import Tensor
from torch.nn import functional as F

import torch

@torch.jit._script_if_tracing
def encode_keyps(reference_keyps: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference landmarks

    Args:
        reference_keyps (Tensor): landmarks to be encoded
        proposals (Tensor): reference boxes
        weights (Tensor[2]): the weights for ``(x, y)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    # reference_keyps = torch.reshape(reference_keyps, (reference_keyps.size(0), 5, 2))

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths  = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x   = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y   = proposals_y1 + 0.5 * ex_heights

    ex_widths  = ex_widths.expand(reference_keyps.size(0), 5).unsqueeze(2)
    ex_heights = ex_heights.expand(reference_keyps.size(0), 5).unsqueeze(2)
    ex_ctr_x   = ex_ctr_x.expand(reference_keyps.size(0), 5).unsqueeze(2)
    ex_ctr_y   = ex_ctr_y.expand(reference_keyps.size(0), 5).unsqueeze(2)
    priors = torch.cat([ex_ctr_x, ex_ctr_y, ex_widths, ex_heights], dim=2)

    targets = reference_keyps - priors[:, :, :2]
    targets /= (weights * priors[:, :, 2:])

    targets = targets.reshape(targets.size(0), -1)
    return targets


class KeypointCoder:
    """
    This class encodes and decodes a set of keypoints into
    the representation used for training the regressors.
    """

    def __init__(
        self, weights: Tuple[float, float]
    ) -> None:
        """
        Args:
            weights (2-element tuple)
        """
        self.weights = weights

    def encode(self, reference_keyps: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        keyps_per_image = [len(b) for b in reference_keyps]
        reference_keyps = torch.cat(reference_keyps, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_keyps, proposals)
        return targets.split(keyps_per_image, 0)

    def encode_single(self, reference_keyps: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_keyps (Tensor): reference boxes
            proposals (Tensor): landmarks to be encoded
        """
        dtype = reference_keyps.dtype
        device = reference_keyps.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_keyps(reference_keyps, proposals, weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        torch._assert(
            isinstance(boxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(rel_codes, torch.Tensor),
            "This function expects rel_codes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_keyps = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_keyps = pred_keyps.reshape(box_sum, -1, 10)
        return pred_keyps

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative landmarks offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded landmarks.
            boxes (Tensor): reference boxes(anchors).
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        priors = torch.stack([ctr_x,ctr_y,widths,heights],dim=1)

        dtype = rel_codes.dtype
        device = rel_codes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)

        pred_keyps = torch.cat((priors[:, :2] + rel_codes[:, :2] * weights * priors[:, 2:],
                        priors[:, :2] + rel_codes[:, 2:4] * weights * priors[:, 2:],
                        priors[:, :2] + rel_codes[:, 4:6] * weights * priors[:, 2:],
                        priors[:, :2] + rel_codes[:, 6:8] * weights * priors[:, 2:],
                        priors[:, :2] + rel_codes[:, 8:10] * weights * priors[:, 2:],
                        ), dim=1)

        return pred_keyps

def _keyp_loss(
    keyp_coder: KeypointCoder,
    anchors_per_image: Tensor,
    matched_gt_keyps_per_image: Tensor,
    keyp_regression_per_image: Tensor,
    cnf: Optional[Dict[str, float]] = None,
) -> Tensor:
    # 只关注人脸关键点
    target_regression = keyp_coder.encode_single(matched_gt_keyps_per_image, anchors_per_image)
    beta = cnf["beta"] if cnf is not None and "beta" in cnf else 1.0
    return F.smooth_l1_loss(keyp_regression_per_image, target_regression, reduction="sum", beta=beta)
    