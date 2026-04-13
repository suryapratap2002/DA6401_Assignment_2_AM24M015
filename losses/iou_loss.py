

import torch
import torch.nn as nn
from typing import Optional


class IoULoss(nn.Module):

    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. Choose from 'mean', 'sum', 'none'."
            )
        self.reduction = reduction
        self.eps = eps

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _compute_iou(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:

        pred_xyxy = self._cxcywh_to_xyxy(pred)
        tgt_xyxy = self._cxcywh_to_xyxy(target)

        # Intersection
        inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h  # (B,)

        # Areas
        pred_area = pred[:, 2].clamp(min=0) * pred[:, 3].clamp(min=0)
        tgt_area = target[:, 2].clamp(min=0) * target[:, 3].clamp(min=0)

        union = pred_area + tgt_area - intersection  # (B,)

        iou = (intersection + self.eps) / (union + self.eps)  # (B,) in [0,1]
        return iou.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor  (B, 4)  –  [x_center, y_center, width, height] predicted
        target : Tensor  (B, 4)  –  ground-truth in the same format
        reduction : str | None
            Override the instance-level reduction for this call. If None,
            uses self.reduction.

        Returns
        -------
        Scalar (or (B,) tensor when reduction="none").
        """
        red = reduction if reduction is not None else self.reduction
        if red not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{red}'.")

        iou = self._compute_iou(pred, target)  # (B,)
        loss = 1.0 - iou                       # (B,) in [0, 1]

        if red == "mean":
            return loss.mean()
        elif red == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, eps={self.eps}"
