import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import UNet


_HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")


def _load_state(path: str):

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                return state[key]
    return state


class MultiTaskPerceptionModel(nn.Module):

    def __init__(
        self,
        num_classes: int = 37,
        num_seg_classes: int = 3,
        device: str = "cpu",
        classifier_ckpt: str = "checkpoints/classifier.pth",
        localizer_ckpt: str  = "checkpoints/localizer.pth",
        unet_ckpt: str       = "checkpoints/unet.pth",
    ):
        super().__init__()
        self.device = torch.device(device)

        # ── Auto-download checkpoints from Google Drive ──────────────────
        import gdown
        _drive_ids = {
            "checkpoints/classifier.pth": "1gnWIfelUJz6WyJQcqlDrq4Wdvkih3fxD",
            "checkpoints/localizer.pth": "1JeuACIdtmUq9GkdnJNOwz21j0IzYrvuB",
            "checkpoints/unet.pth": "1PJ1f2jz335PCnQ5ROOYVzR3iPB8JmcAh",
        }
        os.makedirs(os.path.join(_HERE, "checkpoints"), exist_ok=True)
        for rel, fid in _drive_ids.items():
            dest = os.path.join(_HERE, rel)
            if not os.path.isfile(dest) and not fid.startswith("PASTE_"):
                gdown.download(id=fid, output=dest, quiet=False)


        # 1. Shared VGG11 backbone  (loaded from classifier checkpoint)

        vgg = VGG11(num_classes=num_classes)
        self._try_load(vgg, classifier_ckpt)

        self.backbone_b1 = vgg.block1
        self.backbone_b2 = vgg.block2
        self.backbone_b3 = vgg.block3
        self.backbone_b4 = vgg.block4
        self.backbone_b5 = vgg.block5
        self.avgpool     = vgg.avgpool



        # 2. Classification head  (from classifier checkpoint)

        self.cls_head = vgg.classifier



        # 3. Localisation head  (from localizer checkpoint)


        loc_model = LocalizationModel(vgg=VGG11(num_classes=num_classes))
        self._try_load(loc_model, localizer_ckpt)
        self.reg_head = loc_model.reg_head
        del loc_model

        # 4. Segmentation decoder  (from unet checkpoint)

        unet = UNet(vgg=VGG11(num_classes=num_classes), num_classes=num_seg_classes)
        self._try_load(unet, unet_ckpt)

        self.seg_bridge = unet.bridge
        self.seg_up5    = unet.up5
        self.seg_dec5   = unet.dec5
        self.seg_up4    = unet.up4
        self.seg_dec4   = unet.dec4
        self.seg_up3    = unet.up3
        self.seg_dec3   = unet.dec3
        self.seg_up2    = unet.up2
        self.seg_dec2   = unet.dec2
        self.seg_up1    = unet.up1
        self.seg_dec1   = unet.dec1
        self.seg_head   = unet.head
        del unet

        self.to(self.device)


    def _try_load(self, model: nn.Module, rel_path: str) -> None:
        """Attempt to load checkpoint; warn if not found (use random init)."""
        full = os.path.join(_HERE, rel_path)
        if os.path.isfile(full):
            state = _load_state(full)
            model.load_state_dict(state, strict=False)
            print(f"[MultiTask] Loaded: {rel_path}")
        else:
            print(f"[MultiTask] Not found (random init): {rel_path}")

    @staticmethod
    def _pad_and_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle off-by-one spatial mismatches before concatenation."""
        dH = skip.shape[2] - up.shape[2]
        dW = skip.shape[3] - up.shape[3]
        if dH != 0 or dW != 0:
            import torch.nn.functional as F_pad
            up = F_pad.pad(up, [0, dW, 0, dH])
        return torch.cat([up, skip], dim=1)



    def forward(self, x: torch.Tensor):

        s1 = self.backbone_b1(x)    # (B,  64, 112, 112)
        s2 = self.backbone_b2(s1)   # (B, 128,  56,  56)
        s3 = self.backbone_b3(s2)   # (B, 256,  28,  28)
        s4 = self.backbone_b4(s3)   # (B, 512,  14,  14)
        s5 = self.backbone_b5(s4)   # (B, 512,   7,   7)

        pooled = self.avgpool(s5)   # (B, 512, 7, 7)

        # ---- Task 1: Classification ----
        class_logits = self.cls_head(pooled)

        # ---- Task 2: Localisation ----
        bbox = self.reg_head(pooled)

        # ---- Task 3: Segmentation (UNet decoder) ----
        b  = self.seg_bridge(s5)           # (B, 512,   7,   7)

        d5 = self.seg_up5(b)               # (B, 512,  14,  14)
        d5 = self._pad_and_cat(d5, s4)
        d5 = self.seg_dec5(d5)

        d4 = self.seg_up4(d5)              # (B, 256,  28,  28)
        d4 = self._pad_and_cat(d4, s3)
        d4 = self.seg_dec4(d4)

        d3 = self.seg_up3(d4)              # (B, 128,  56,  56)
        d3 = self._pad_and_cat(d3, s2)
        d3 = self.seg_dec3(d3)

        d2 = self.seg_up2(d3)              # (B,  64, 112, 112)
        d2 = self._pad_and_cat(d2, s1)
        d2 = self.seg_dec2(d2)

        d1 = self.seg_up1(d2)              # (B,  32, 224, 224)
        d1 = self.seg_dec1(d1)
        seg_logits = self.seg_head(d1)     # (B, num_seg_classes, 224, 224)

        return class_logits, bbox, seg_logits
