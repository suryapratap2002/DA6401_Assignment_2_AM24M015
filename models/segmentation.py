"""
U-Net style Semantic Segmentation model (Task 3).

Contracting path  : VGG11 five-block encoder with skip-connection outputs.
Expansive path    : Symmetric decoder using *Transposed Convolutions* for
                    upsampling (bilinear / unpooling are explicitly forbidden).

Architecture
------------
Encoder outputs (after each MaxPool):
  block1 → (B, 64,  112, 112)
  block2 → (B, 128,  56,  56)
  block3 → (B, 256,  28,  28)
  block4 → (B, 512,  14,  14)
  block5 → (B, 512,   7,   7)

Bridge (no upsampling, deepest feature):
  → 2×Conv(512→512) → (B, 512, 7, 7)

Decoder:
  Up5 : ConvTranspose2d(512→512, k=2, s=2) → cat(skip4=512) → 2×Conv(1024→512)
  Up4 : ConvTranspose2d(512→256, k=2, s=2) → cat(skip3=256) → 2×Conv(512→256)
  Up3 : ConvTranspose2d(256→128, k=2, s=2) → cat(skip2=128) → 2×Conv(256→128)
  Up2 : ConvTranspose2d(128→64,  k=2, s=2) → cat(skip1=64)  → 2×Conv(128→64)
  Up1 : ConvTranspose2d(64→32,   k=2, s=2)                  → 2×Conv(32→32)

Head : Conv1×1(32 → num_classes)

Loss
----
Dice loss (+ optional cross-entropy weighting).  The Oxford-IIIT Pet trimap
has 3 classes: 1=foreground, 2=background, 3=boundary.  Dice is chosen as
the primary loss because it handles class imbalance better than pixel-wise
cross-entropy: the boundary class is far fewer pixels but critically important
for contour quality.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    U-Net segmentation model with VGG11 encoder.

    Parameters
    ----------
    vgg : VGG11 | None
        Pre-trained VGG11 instance. If None a new one is created.
    num_classes : int
        Number of segmentation classes. Default: 3 (Pet trimap).
    freeze_encoder : bool
        If True, VGG11 backbone parameters are frozen. Default: False.
    """

    def __init__(
        self,
        vgg: VGG11 = None,
        num_classes: int = 3,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        if vgg is None:
            vgg = VGG11()

        # ---- Encoder (VGG11 blocks, no MaxPool on the last one for bridge) ----
        self.enc1 = vgg.block1   # out: 64 ch, ½ spatial
        self.enc2 = vgg.block2   # out: 128 ch, ¼
        self.enc3 = vgg.block3   # out: 256 ch, ⅛
        self.enc4 = vgg.block4   # out: 512 ch, 1/16
        self.enc5 = vgg.block5   # out: 512 ch, 1/32

        if freeze_encoder:
            for p in list(self.enc1.parameters()) + list(self.enc2.parameters()) + \
                     list(self.enc3.parameters()) + list(self.enc4.parameters()) + \
                     list(self.enc5.parameters()):
                p.requires_grad = False

        # ---- Bridge ----
        self.bridge = _double_conv(512, 512)

        # ---- Decoder (transposed convolutions + skip connections) ----
        # Up5: 7→14, concat with enc4 output (512 ch)
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)

        # Up4: 14→28, concat with enc3 output (256 ch)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 256, 256)

        # Up3: 28→56, concat with enc2 output (128 ch)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 128, 128)

        # Up2: 56→112, concat with enc1 output (64 ch)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 64, 64)

        # Up1: 112→224 (restore full resolution)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _double_conv(32, 32)

        # Segmentation head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        # Init decoder weights
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        decoder_modules = [
            self.bridge, self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.head,
            self.up5, self.up4, self.up3, self.up2, self.up1,
        ]
        for m_group in decoder_modules:
            for m in (m_group.modules() if hasattr(m_group, 'modules') else [m_group]):
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 3, 224, 224)

        Returns
        -------
        logits : Tensor  (B, num_classes, 224, 224)
        """
        # Encoder with skip connections
        # VGG blocks include MaxPool, so spatial dims halve at each block
        s1 = self.enc1(x)   # (B,  64, 112, 112)
        s2 = self.enc2(s1)  # (B, 128,  56,  56)
        s3 = self.enc3(s2)  # (B, 256,  28,  28)
        s4 = self.enc4(s3)  # (B, 512,  14,  14)
        s5 = self.enc5(s4)  # (B, 512,   7,   7)

        # Bridge
        b = self.bridge(s5)  # (B, 512, 7, 7)

        # Decoder
        d5 = self.up5(b)                                # (B, 512, 14, 14)
        d5 = self._pad_and_cat(d5, s4)                  # cat skip from enc4
        d5 = self.dec5(d5)                              # (B, 512, 14, 14)

        d4 = self.up4(d5)                               # (B, 256, 28, 28)
        d4 = self._pad_and_cat(d4, s3)                  # cat skip from enc3
        d4 = self.dec4(d4)                              # (B, 256, 28, 28)

        d3 = self.up3(d4)                               # (B, 128, 56, 56)
        d3 = self._pad_and_cat(d3, s2)                  # cat skip from enc2
        d3 = self.dec3(d3)                              # (B, 128, 56, 56)

        d2 = self.up2(d3)                               # (B,  64, 112, 112)
        d2 = self._pad_and_cat(d2, s1)                  # cat skip from enc1
        d2 = self.dec2(d2)                              # (B,  64, 112, 112)

        d1 = self.up1(d2)                               # (B,  32, 224, 224)
        d1 = self.dec1(d1)

        return self.head(d1)                            # (B, num_classes, 224, 224)

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Handle any off-by-one spatial mismatches that can arise when the
        input spatial size is not a perfect power of 2.
        """
        dH = skip.shape[2] - upsampled.shape[2]
        dW = skip.shape[3] - upsampled.shape[3]
        if dH != 0 or dW != 0:
            import torch.nn.functional as F
            upsampled = F.pad(upsampled, [0, dW, 0, dH])
        return torch.cat([upsampled, skip], dim=1)
