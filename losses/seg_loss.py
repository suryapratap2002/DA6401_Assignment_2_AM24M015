
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):


    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                     # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        targets_oh = F.one_hot(targets.long(), num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()

        # Flatten spatial dims
        probs_flat = probs.view(probs.shape[0], num_classes, -1)      # (B,C,N)
        tgt_flat   = targets_oh.view(targets_oh.shape[0], num_classes, -1)

        intersection = (probs_flat * tgt_flat).sum(dim=2)   # (B, C)
        cardinality  = probs_flat.sum(dim=2) + tgt_flat.sum(dim=2)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class   # (B, C)

        if self.reduction == "mean":
            return dice_loss.mean()
        return dice_loss.sum()


class CombinedSegLoss(nn.Module):

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + \
               self.ce_weight   * self.ce(logits, targets.long())
