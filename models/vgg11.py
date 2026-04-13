import torch
import torch.nn as nn
from models.layers import CustomDropout


def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class VGG11(nn.Module):

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # ---- Feature Extractor (5 convolutional blocks) ----
        # Block 1
        self.block1 = nn.Sequential(
            _make_conv_block(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2
        self.block2 = nn.Sequential(
            _make_conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3
        self.block3 = nn.Sequential(
            _make_conv_block(128, 256),
            _make_conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4
        self.block4 = nn.Sequential(
            _make_conv_block(256, 512),
            _make_conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 5
        self.block5 = nn.Sequential(
            _make_conv_block(512, 512),
            _make_conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (as per Simonyan et al. supplementary)
    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Backbone accessor – returns feature-map blocks for downstream tasks
    # ------------------------------------------------------------------
    def get_backbone(self) -> nn.ModuleList:
        """Return the five convolutional blocks as a ModuleList."""
        return nn.ModuleList([
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward_features(self, x: torch.Tensor):
        """
        Return intermediate feature maps from each block.
        Used by the localizer encoder and U-Net skip connections.

        Returns
        -------
        list of Tensors: [b1, b2, b3, b4, b5]
        """
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)
        return [b1, b2, b3, b4, b5]
