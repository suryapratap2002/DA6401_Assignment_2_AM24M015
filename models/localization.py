"""
Object Localization model (Task 2).

Uses the VGG11 convolutional backbone as an encoder and attaches a
lightweight regression head that predicts [x_center, y_center, width, height]
in *pixel* coordinates (not normalised).

Encoder strategy
----------------
We offer a `freeze_backbone` flag (default False = fine-tuning).

Justification: The Oxford-IIIT Pet dataset is moderately different from
ImageNet – the bounding box annotations mark *head* regions so fine-grained
texture features matter.  Empirically, fine-tuning the last two conv blocks
(or all blocks with a low LR) consistently improves localisation compared to
a frozen backbone.  The default therefore unfreezes all parameters, but the
flag is provided for ablation studies.

Output head
-----------
After AdaptiveAvgPool2d(7,7) the spatial feature volume is flattened to
512*7*7=25088 → FC(1024) → ReLU → FC(4) → ReLU (ensures non-negative
pixel-space values for width/height; x_c and y_c are also non-negative for
a 224×224 image).
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


class LocalizationModel(nn.Module):
    """
    VGG11-based single-object localisation model.

    Parameters
    ----------
    vgg : VGG11 | None
        Pre-trained (or fresh) VGG11 instance.  If None a new VGG11 is
        created.
    freeze_backbone : bool
        If True, all convolutional parameters are frozen and only the
        regression head is trained.  Default: False (fine-tune all).
    image_size : int
        Assumed input spatial size (square).  Used to initialise the
        regression head's output range. Default: 224.
    """

    def __init__(
        self,
        vgg: VGG11 = None,
        freeze_backbone: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size

        if vgg is None:
            vgg = VGG11()

        # Take only the 5 conv blocks from VGG11
        self.block1 = vgg.block1
        self.block2 = vgg.block2
        self.block3 = vgg.block3
        self.block4 = vgg.block4
        self.block5 = vgg.block5
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_backbone:
            for param in self._backbone_params():
                param.requires_grad = False

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
            nn.ReLU(inplace=True),   # bounding-box values are always ≥ 0
        )

        # Initialise the head
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _backbone_params(self):
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            yield from block.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 3, 224, 224)

        Returns
        -------
        Tensor  (B, 4)  –  [x_center, y_center, width, height] in pixel space
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        return self.reg_head(x)
