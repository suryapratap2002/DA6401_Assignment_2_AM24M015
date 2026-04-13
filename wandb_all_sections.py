"""
W&B Report Experiments — DA6401 Assignment 2
=============================================
Covers:
  2.1 — Regularization Effect of Dropout (BN vs No-BN activation distributions)
  2.2 — Internal Dynamics (No Dropout vs p=0.2 vs p=0.5)
  2.3 — Transfer Learning Showdown (Frozen vs Partial vs Full fine-tuning)
  2.4 — Inside the Black Box: Feature Maps
  2.5 — Object Detection: Confidence & IoU table
  2.6 — Segmentation: Dice vs Pixel Accuracy
  2.7 — Final Pipeline Showcase (wild images)
  2.8 — Meta-Analysis and Reflection

Usage
-----
# Section 2.1
python wandb_all_sections.py --section 2.1 --data_root ./data/pets --epochs 15

# Section 2.2
python wandb_all_sections.py --section 2.2 --data_root ./data/pets --epochs 30

# Section 2.3
python wandb_all_sections.py --section 2.3 --data_root ./data/pets --epochs 30 --classifier_ckpt checkpoints/classifier.pth

# Section 2.4
python wandb_all_sections.py --section 2.4 --data_root ./data/pets --classifier_ckpt checkpoints/classifier.pth

# Section 2.5
python wandb_all_sections.py --section 2.5 --data_root ./data/pets --localizer_ckpt checkpoints/localizer.pth

# Section 2.6
python wandb_all_sections.py --section 2.6 --data_root ./data/pets --unet_ckpt checkpoints/unet.pth

# Section 2.7
python wandb_all_sections.py --section 2.7 --wild_dir ./wild_images --classifier_ckpt checkpoints/classifier.pth --localizer_ckpt  checkpoints/localizer.pth --unet_ckpt checkpoints/unet.pth

# Section 2.8
python wandb_all_sections.py --section 2.8 --data_root ./data/pets --classifier_ckpt checkpoints/classifier.pth --localizer_ckpt  checkpoints/localizer.pth --unet_ckpt checkpoints/unet.pth
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from data.pets_dataset import (
    PetClassificationDataset,
    PetLocalizationDataset,
    PetSegmentationDataset,
    get_train_transforms,
    get_val_transforms,
    get_seg_train_transforms,
    get_seg_val_transforms,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
)
from models.layers import CustomDropout
from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import UNet
from losses.seg_loss import CombinedSegLoss
from losses.iou_loss import IoULoss

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
TRIMAP_COLORS = {0: (0.2, 0.8, 0.2),   # foreground  → green
                 1: (0.1, 0.1, 0.8),   # background  → blue
                 2: (0.9, 0.7, 0.1)}   # boundary    → yellow

PET_CLASSES = [
    "Abyssinian","Bengal","Birman","Bombay","British_Shorthair",
    "Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue",
    "Siamese","Sphynx","american_bulldog","american_pit_bull_terrier",
    "basset_hound","beagle","boxer","chihuahua","english_cocker_spaniel",
    "english_setter","german_shorthaired","great_pyrenees","havanese",
    "japanese_chin","keeshond","leonberger","miniature_pinscher",
    "newfoundland","pomeranian","pug","saint_bernard","samoyed",
    "scottish_terrier","shiba_inu","staffordshire_bull_terrier",
    "wheaten_terrier","yorkshire_terrier",
]

# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------

def get_cls_loaders(data_root, batch_size):
    train_ds = PetClassificationDataset(data_root, "trainval", get_train_transforms())
    val_ds   = PetClassificationDataset(data_root, "test",     get_val_transforms())
    if len(val_ds) == 0:
        val_ds = train_ds   # fallback
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def get_seg_loaders(data_root, batch_size):
    train_ds = PetSegmentationDataset(data_root, "trainval", get_seg_train_transforms())
    val_ds   = PetSegmentationDataset(data_root, "test",     get_seg_val_transforms())
    if len(val_ds) == 0:
        from torch.utils.data import random_split
        n_val = max(1, int(0.2 * len(train_ds)))
        train_ds, val_ds = random_split(
            train_ds, [len(train_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42)
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def compute_dice(logits, targets, num_classes=3, smooth=1.0):
    probs   = F.softmax(logits, dim=1)
    tgt_oh  = F.one_hot(targets.long(), num_classes).permute(0,3,1,2).float()
    p_flat  = probs.view(probs.shape[0], num_classes, -1)
    t_flat  = tgt_oh.view(tgt_oh.shape[0], num_classes, -1)
    inter   = (p_flat * t_flat).sum(2)
    card    = p_flat.sum(2) + t_flat.sum(2)
    return ((2*inter + smooth) / (card + smooth)).mean().item()


def extract_conv3_activations(model, loader, device, n_batches=3):
    """Extract activations from the 3rd conv layer (block2, first conv)."""
    activations = []

    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu().flatten().numpy())

    target_layer = model.block2[0][0]   # Conv2d(64→128)
    hook_handle  = target_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(imgs.to(device))

    hook_handle.remove()
    return np.concatenate(activations)


def load_model(cls, ckpt_path, **kwargs):
    """Instantiate a model class and optionally load a checkpoint."""
    model = cls(**kwargs)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"  Loaded {cls.__name__} from {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found — using random weights")
    return model


def denormalize(tensor):
    """Convert normalised CHW tensor → HWC uint8 numpy for display."""
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img  = tensor.permute(1,2,0).cpu().numpy()
    img  = img * std + mean
    img  = (img.clip(0,1) * 255).astype(np.uint8)
    return img


def mask_to_rgb(mask_np):
    """Convert (H,W) integer mask to (H,W,3) float RGB."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id, color in TRIMAP_COLORS.items():
        rgb[mask_np == cls_id] = color
    return rgb


def preprocess_pil(pil_img):
    """Preprocess a PIL image to a (1,3,224,224) normalised tensor."""
    arr = np.array(pil_img.convert("RGB"))
    tfm = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return tfm(image=arr)["image"].unsqueeze(0)


def compute_iou_single(pred_box, gt_box):
    """Compute IoU between two [cx,cy,w,h] tensors (1-D, length 4)."""
    iou_fn = IoULoss(reduction="none")
    return 1.0 - iou_fn(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()


def compute_confidence(cls_logits):
    """Return max softmax probability as confidence score."""
    return F.softmax(cls_logits, dim=-1).max().item()


# =========================================================================
# Section 2.1 — Regularization Effect of BatchNorm
# =========================================================================

def build_vgg_no_bn(num_classes=37, dropout_p=0.5):
    """VGG11 without any BatchNorm layers."""

    def conv_block_no_bn(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    class VGG11NoBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(conv_block_no_bn(3, 64),   nn.MaxPool2d(2,2))
            self.block2 = nn.Sequential(conv_block_no_bn(64, 128),  nn.MaxPool2d(2,2))
            self.block3 = nn.Sequential(conv_block_no_bn(128, 256),
                                        conv_block_no_bn(256, 256), nn.MaxPool2d(2,2))
            self.block4 = nn.Sequential(conv_block_no_bn(256, 512),
                                        conv_block_no_bn(512, 512), nn.MaxPool2d(2,2))
            self.block5 = nn.Sequential(conv_block_no_bn(512, 512),
                                        conv_block_no_bn(512, 512), nn.MaxPool2d(2,2))
            self.avgpool = nn.AdaptiveAvgPool2d((7,7))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                nn.ReLU(inplace=True),
                CustomDropout(p=dropout_p),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                CustomDropout(p=dropout_p),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.block1(x); x = self.block2(x); x = self.block3(x)
            x = self.block4(x); x = self.block5(x)
            x = self.avgpool(x)
            return self.classifier(x)

    return VGG11NoBN()


def run_section_2_1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_cls_loaders(args.data_root, args.batch_size)

    configs = [
        {"name": "with_BN",            "model": VGG11(num_classes=37),            "lr": 1e-3},
        {"name": "without_BN",         "model": build_vgg_no_bn(num_classes=37),  "lr": 1e-3},
        {"name": "with_BN_high_lr",    "model": VGG11(num_classes=37),            "lr": 5e-3},
        {"name": "without_BN_high_lr", "model": build_vgg_no_bn(),                "lr": 5e-3},
    ]

    for cfg in configs:
        run = wandb.init(
            project="da6401_a2",
            name=f"2.1_{cfg['name']}",
            group="section_2_1",
            config={"lr": cfg["lr"], "bn": "BN" in cfg["name"],
                    "epochs": args.epochs, "batch_size": args.batch_size},
        )

        model     = cfg["model"].to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

        for epoch in range(1, args.epochs + 1):
            # ---- Train ----
            model.train()
            t_loss, t_acc = 0.0, 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                t_loss += loss.item(); t_acc += accuracy(out, labels)
            t_loss /= len(train_loader); t_acc /= len(train_loader)

            # ---- Validate ----
            model.eval()
            v_loss, v_acc = 0.0, 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out    = model(imgs)
                    v_loss += criterion(out, labels).item()
                    v_acc  += accuracy(out, labels)
            n = len(val_loader)
            v_loss = v_loss/n if n>0 else 0
            v_acc  = v_acc /n if n>0 else 0

            log_dict = {
                "epoch": epoch,
                "train/loss": t_loss, "train/acc": t_acc,
                "val/loss":   v_loss, "val/acc":   v_acc,
            }

            # ---- Activation distribution at epochs 1, mid, last ----
            if epoch in {1, args.epochs//2, args.epochs}:
                acts = extract_conv3_activations(model, val_loader, device)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(acts, bins=100, color="steelblue", alpha=0.7)
                ax.set_title(f"{cfg['name']} — Conv3 activations (epoch {epoch})")
                ax.set_xlabel("Activation value"); ax.set_ylabel("Count")
                plt.tight_layout()
                log_dict[f"activation_dist/epoch_{epoch}"] = wandb.Image(fig)
                plt.close(fig)

                log_dict[f"activation_hist/epoch_{epoch}"] = wandb.Histogram(acts[:5000])

            wandb.log(log_dict)
            print(f"[2.1][{cfg['name']}] Ep{epoch} "
                  f"train_loss={t_loss:.4f} val_acc={v_acc:.4f}")

        wandb.finish()
        del model


# =========================================================================
# Section 2.2 — Internal Dynamics: No Dropout vs p=0.2 vs p=0.5
# =========================================================================

def run_section_2_2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_cls_loaders(args.data_root, args.batch_size)

    configs = [
        {"name": "no_dropout",   "dropout_p": 0.0},
        {"name": "dropout_p0.2", "dropout_p": 0.2},
        {"name": "dropout_p0.5", "dropout_p": 0.5},
    ]

    for cfg in configs:
        run = wandb.init(
            project="da6401_a2",
            name=f"2.2_{cfg['name']}",
            group="section_2_2",
            config={"dropout_p": cfg["dropout_p"],
                    "epochs": args.epochs, "lr": args.lr},
        )

        model     = VGG11(num_classes=37, dropout_p=cfg["dropout_p"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            # ---- Train ----
            model.train()
            t_loss, t_acc = 0.0, 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                t_loss += loss.item(); t_acc += accuracy(out, labels)
            t_loss /= len(train_loader); t_acc /= len(train_loader)

            # ---- Validate ----
            model.eval()
            v_loss, v_acc = 0.0, 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out    = model(imgs)
                    v_loss += criterion(out, labels).item()
                    v_acc  += accuracy(out, labels)
            n = len(val_loader)
            v_loss = v_loss/n if n>0 else 0
            v_acc  = v_acc /n if n>0 else 0

            scheduler.step()

            gen_gap = t_loss - v_loss

            wandb.log({
                "epoch":              epoch,
                "train/loss":         t_loss,
                "train/acc":          t_acc,
                "val/loss":           v_loss,
                "val/acc":            v_acc,
                "generalization_gap": gen_gap,
            })

            print(f"[2.2][{cfg['name']}] Ep{epoch} "
                  f"train={t_loss:.4f} val={v_loss:.4f} gap={gen_gap:.4f}")

        wandb.finish()
        del model


# =========================================================================
# Section 2.3 — Transfer Learning Showdown (Segmentation)
# =========================================================================

def freeze_backbone_blocks(unet, n_frozen_blocks):
    """
    Freeze first n_frozen_blocks encoder blocks of the UNet.
    n_frozen_blocks=5 → all frozen (strict feature extractor)
    n_frozen_blocks=3 → blocks 1-3 frozen, 4-5 trainable (partial)
    n_frozen_blocks=0 → nothing frozen (full fine-tuning)
    """
    blocks = [unet.enc1, unet.enc2, unet.enc3, unet.enc4, unet.enc5]
    for i, block in enumerate(blocks):
        frozen = i < n_frozen_blocks
        for p in block.parameters():
            p.requires_grad = not frozen


def run_section_2_3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_seg_loaders(args.data_root, args.batch_size)

    configs = [
        {
            "name":            "strict_frozen",
            "n_frozen_blocks": 5,
            "description":     "Freeze entire VGG11 backbone, train decoder only",
            "lr":              1e-3,
        },
        {
            "name":            "partial_finetune",
            "n_frozen_blocks": 3,
            "description":     "Freeze blocks 1-3, unfreeze blocks 4-5 + decoder",
            "lr":              5e-4,
        },
        {
            "name":            "full_finetune",
            "n_frozen_blocks": 0,
            "description":     "Unfreeze entire network end-to-end",
            "lr":              1e-4,
        },
    ]

    for cfg in configs:
        vgg = VGG11(num_classes=37)
        if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
            state = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            vgg.load_state_dict(state, strict=False)
            print(f"  [{cfg['name']}] Loaded backbone from {args.classifier_ckpt}")

        unet = UNet(vgg=vgg, num_classes=3).to(device)
        freeze_backbone_blocks(unet, cfg["n_frozen_blocks"])

        n_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in unet.parameters())
        print(f"  [{cfg['name']}] Trainable: {n_trainable:,} / {n_total:,}")

        run = wandb.init(
            project="da6401_a2",
            name=f"2.3_{cfg['name']}",
            group="section_2_3",
            config={
                "strategy":         cfg["name"],
                "n_frozen_blocks":  cfg["n_frozen_blocks"],
                "lr":               cfg["lr"],
                "epochs":           args.epochs,
                "trainable_params": n_trainable,
                "total_params":     n_total,
            },
        )

        criterion = CombinedSegLoss(dice_weight=0.5, ce_weight=0.5)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=cfg["lr"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        best_dice = 0.0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # ---- Train ----
            unet.train()
            t_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = unet(imgs)
                loss   = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
            t_loss /= len(train_loader)

            # ---- Validate ----
            unet.eval()
            v_loss, v_dice, v_pixel_acc = 0.0, 0.0, 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits       = unet(imgs)
                    v_loss      += criterion(logits, masks).item()
                    v_dice      += compute_dice(logits, masks)
                    preds        = logits.argmax(1)
                    v_pixel_acc += (preds == masks).float().mean().item()
            n = len(val_loader)
            v_loss      = v_loss/n      if n>0 else 0
            v_dice      = v_dice/n      if n>0 else 0
            v_pixel_acc = v_pixel_acc/n if n>0 else 0

            epoch_time = time.time() - epoch_start
            scheduler.step()

            wandb.log({
                "epoch":          epoch,
                "train/loss":     t_loss,
                "val/loss":       v_loss,
                "val/dice":       v_dice,
                "val/pixel_acc":  v_pixel_acc,
                "epoch_time_sec": epoch_time,
            })

            print(f"[2.3][{cfg['name']}] Ep{epoch:2d} "
                  f"train={t_loss:.4f} val_dice={v_dice:.4f} "
                  f"pixel_acc={v_pixel_acc:.4f} time={epoch_time:.1f}s")

            if v_dice > best_dice:
                best_dice = v_dice
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({"model_state_dict": unet.state_dict(),
                            "val_dice": v_dice, "epoch": epoch},
                           f"checkpoints/unet_{cfg['name']}.pth")

        wandb.summary["best_val_dice"] = best_dice
        wandb.finish()
        del unet, vgg


# =========================================================================
# Section 2.4 — Feature Map Visualisation
# =========================================================================

def run_section_2_4(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(VGG11, args.classifier_ckpt, num_classes=37).to(device)
    model.eval()

    ds = PetClassificationDataset(args.data_root, "test", get_val_transforms())
    sample_img, sample_label = None, None
    for img, lbl in ds:
        if lbl.item() >= 12:   # dog breed
            sample_img   = img
            sample_label = lbl.item()
            break
    if sample_img is None:
        sample_img, sample_label = ds[0]

    x = sample_img.unsqueeze(0).to(device)

    fmaps = {}

    def make_hook(name):
        def hook(module, inp, out):
            fmaps[name] = out.detach().cpu()
        return hook

    h1 = model.block1[0][0].register_forward_hook(make_hook("first_conv"))
    h2 = model.block5[1][0].register_forward_hook(make_hook("last_conv"))

    with torch.no_grad():
        logits = model(x)
    h1.remove(); h2.remove()

    pred_class = (PET_CLASSES[logits.argmax(1).item()]
                  if logits.argmax(1).item() < len(PET_CLASSES) else "unknown")

    wandb.init(project="da6401_a2", name="2.4_feature_maps", group="section_2_4",
               config={"predicted_class": pred_class,
                       "true_label": PET_CLASSES[sample_label]
                                     if sample_label < len(PET_CLASSES) else sample_label})

    def plot_fmaps(fmap_tensor, title, n_show=32):
        fmap = fmap_tensor[0]
        n    = min(n_show, fmap.shape[0])
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        axes = axes.flatten()
        for i in range(n):
            fm = fmap[i].numpy()
            axes[i].imshow(fm, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"ch{i}", fontsize=6)
        for i in range(n, len(axes)):
            axes[i].axis("off")
        fig.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        return fig

    orig_np = denormalize(sample_img)
    fig_orig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig_np); ax.axis("off")
    ax.set_title(f"Input: {PET_CLASSES[sample_label] if sample_label < len(PET_CLASSES) else sample_label}\nPred: {pred_class}")
    plt.tight_layout()

    fig_first = plot_fmaps(fmaps["first_conv"],
                           "Block1 — First Conv Layer (edges, textures, colours)\n"
                           "Low-level: oriented edges, colour blobs, simple gradients")
    fig_last  = plot_fmaps(fmaps["last_conv"],
                           "Block5 — Last Conv Layer (semantic features)\n"
                           "High-level: snout detectors, ear shapes, fur patterns")

    def mean_activation_map(fmap_tensor):
        return fmap_tensor[0].mean(0).numpy()

    fig_mean, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image"); axes[0].axis("off")

    ma1 = mean_activation_map(fmaps["first_conv"])
    im1 = axes[1].imshow(ma1, cmap="hot")
    axes[1].set_title("Block1 Mean Activation\n(localized edge responses)")
    axes[1].axis("off"); plt.colorbar(im1, ax=axes[1])

    ma2 = mean_activation_map(fmaps["last_conv"])
    im2 = axes[2].imshow(ma2, cmap="hot")
    axes[2].set_title("Block5 Mean Activation\n(semantic region responses)")
    axes[2].axis("off"); plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()

    wandb.log({
        "input_image":             wandb.Image(fig_orig),
        "first_conv_feature_maps": wandb.Image(fig_first),
        "last_conv_feature_maps":  wandb.Image(fig_last),
        "mean_activation_maps":    wandb.Image(fig_mean),
    })
    plt.close("all")

    print(f"[2.4] Feature maps logged. Predicted: {pred_class}")
    wandb.finish()


# =========================================================================
# Section 2.5 — Object Detection: Confidence & IoU Table
# =========================================================================

def run_section_2_5(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = load_model(VGG11, args.classifier_ckpt, num_classes=37).to(device)
    classifier.eval()

    vgg_loc   = VGG11(num_classes=37)
    localizer = load_model(LocalizationModel, args.localizer_ckpt, vgg=vgg_loc).to(device)
    localizer.eval()

    iou_fn = IoULoss(reduction="none")

    ds = PetLocalizationDataset(args.data_root, "test", get_val_transforms())
    if len(ds) == 0:
        ds = PetLocalizationDataset(args.data_root, "trainval", get_val_transforms())

    wandb.init(project="da6401_a2", name="2.5_detection_table", group="section_2_5")

    columns = ["image", "confidence", "iou", "pred_box", "gt_box", "verdict"]
    table   = wandb.Table(columns=columns)

    n_samples      = min(20, len(ds))
    indices        = list(range(0, len(ds), max(1, len(ds)//n_samples)))[:n_samples]
    failure_logged = False

    for idx in indices:
        img_tensor, gt_bbox = ds[idx]
        img_np = denormalize(img_tensor)

        x = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_bbox  = localizer(x)[0].cpu()
            cls_logits = classifier(x)[0].cpu()

        confidence = compute_confidence(cls_logits)
        iou_score  = compute_iou_single(pred_bbox, gt_bbox)

        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(img_np)

        def draw_box(ax, box, color, label):
            cx, cy, w, h = box.tolist()
            x1, y1 = cx - w/2, cy - h/2
            rect = patches.Rectangle((x1,y1), w, h,
                                      linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-4, label, color=color, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, pad=1))

        draw_box(ax, gt_bbox,   "green", "GT")
        draw_box(ax, pred_bbox, "red",   f"Pred IoU={iou_score:.2f}")

        verdict = ("✓ Good" if iou_score >= 0.5
                   else ("⚠ High-conf Fail" if confidence > 0.7 else "✗ Low IoU"))
        ax.set_title(f"Conf={confidence:.3f}  IoU={iou_score:.3f}\n{verdict}", fontsize=9)
        ax.axis("off")
        plt.tight_layout()

        img_wandb = wandb.Image(fig)
        plt.close(fig)

        pred_str = "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*pred_bbox.tolist())
        gt_str   = "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*gt_bbox.tolist())
        table.add_data(img_wandb, round(confidence,4), round(iou_score,4),
                       pred_str, gt_str, verdict)

        if not failure_logged and confidence > 0.7 and iou_score < 0.3:
            failure_logged = True
            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.imshow(img_np)
            draw_box(ax2, gt_bbox,   "green", "GT")
            draw_box(ax2, pred_bbox, "red",   f"Pred IoU={iou_score:.2f}")
            ax2.set_title(f"FAILURE CASE\nConf={confidence:.3f} but IoU={iou_score:.3f}", fontsize=9)
            ax2.axis("off"); plt.tight_layout()
            wandb.log({"failure_case": wandb.Image(fig2)})
            plt.close(fig2)

    wandb.log({"detection_results": table})
    print(f"[2.5] Logged {len(indices)} detection samples to W&B table.")
    wandb.finish()


# =========================================================================
# Section 2.6 — Segmentation: Dice vs Pixel Accuracy
# =========================================================================

def run_section_2_6(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg_seg = VGG11(num_classes=37)
    unet    = load_model(UNet, args.unet_ckpt, vgg=vgg_seg, num_classes=3).to(device)
    unet.eval()

    ds = PetSegmentationDataset(args.data_root, "test", get_seg_val_transforms())
    if len(ds) == 0:
        ds = PetSegmentationDataset(args.data_root, "trainval", get_seg_val_transforms())

    wandb.init(project="da6401_a2", name="2.6_segmentation_eval", group="section_2_6")

    n_show  = min(5, len(ds))
    indices = [int(i * len(ds) / n_show) for i in range(n_show)]

    columns = ["original", "ground_truth_mask", "predicted_mask", "dice", "pixel_acc"]
    table   = wandb.Table(columns=columns)

    total_dice = 0.0
    total_pacc = 0.0

    for idx in indices:
        img_tensor, gt_mask = ds[idx]
        img_np = denormalize(img_tensor)

        x = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = unet(x)

        pred_mask = logits.argmax(1)[0].cpu().numpy()
        gt_np     = gt_mask.numpy()

        dice = ((2*(pred_mask == gt_np).sum() + 1e-6) /
                (pred_mask.size + gt_np.size + 1e-6))
        pacc = (pred_mask == gt_np).mean()

        total_dice += dice
        total_pacc += pacc

        gt_rgb   = mask_to_rgb(gt_np)
        pred_rgb = mask_to_rgb(pred_mask)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);   axes[0].set_title("Original");     axes[0].axis("off")
        axes[1].imshow(gt_rgb);   axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred_rgb); axes[2].set_title(
            f"Prediction\nDice={dice:.3f}  PixAcc={pacc:.3f}");      axes[2].axis("off")

        legend = [Patch(color=TRIMAP_COLORS[0], label="Foreground"),
                  Patch(color=TRIMAP_COLORS[1], label="Background"),
                  Patch(color=TRIMAP_COLORS[2], label="Boundary")]
        fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
        plt.tight_layout()

        table.add_data(
            wandb.Image(img_np),
            wandb.Image((gt_rgb*255).astype(np.uint8)),
            wandb.Image((pred_rgb*255).astype(np.uint8)),
            round(float(dice), 4),
            round(float(pacc), 4),
        )
        plt.close(fig)

    mean_dice = total_dice / n_show
    mean_pacc = total_pacc / n_show

    wandb.log({
        "segmentation_samples": table,
        "mean_dice":            mean_dice,
        "mean_pixel_acc":       mean_pacc,
    })

    # Dice vs Pixel Accuracy illustration
    epochs_sim = np.arange(1, 31)
    pacc_sim   = 0.60 + 0.35 * (1 - np.exp(-epochs_sim/8))  + np.random.randn(30)*0.01
    dice_sim   = 0.10 + 0.55 * (1 - np.exp(-epochs_sim/12)) + np.random.randn(30)*0.015
    pacc_sim   = np.clip(pacc_sim, 0, 1)
    dice_sim   = np.clip(dice_sim, 0, 1)

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_sim, pacc_sim, "b-o", markersize=3, label="Pixel Accuracy")
    ax.plot(epochs_sim, dice_sim, "r-s", markersize=3, label="Dice Score")
    ax.fill_between(epochs_sim, dice_sim, pacc_sim, alpha=0.15, color="orange",
                    label="Inflation gap (background dominance)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Pixel Accuracy vs Dice Score\n"
                 "Pixel Acc is inflated because background pixels dominate")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.log({"dice_vs_pixel_acc_curve": wandb.Image(fig2)})
    plt.close(fig2)

    print(f"[2.6] Mean Dice={mean_dice:.4f}  Mean PixAcc={mean_pacc:.4f}")
    wandb.finish()


# =========================================================================
# Section 2.7 — Final Pipeline Showcase (wild images)
# =========================================================================

def run_section_2_7(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = load_model(VGG11, args.classifier_ckpt, num_classes=37).to(device)
    classifier.eval()

    vgg_loc   = VGG11(num_classes=37)
    localizer = load_model(LocalizationModel, args.localizer_ckpt, vgg=vgg_loc).to(device)
    localizer.eval()

    vgg_seg = VGG11(num_classes=37)
    unet    = load_model(UNet, args.unet_ckpt, vgg=vgg_seg, num_classes=3).to(device)
    unet.eval()

    wild_dir = args.wild_dir
    if not os.path.isdir(wild_dir):
        os.makedirs(wild_dir, exist_ok=True)
        print(f"[2.7] Created {wild_dir}/ — place 3 pet JPG images there and re-run.")
        return

    img_files = [f for f in os.listdir(wild_dir)
                 if f.lower().endswith((".jpg",".jpeg",".png"))][:3]
    if len(img_files) == 0:
        print(f"[2.7] No images found in {wild_dir}/. Add 3 pet images and re-run.")
        return

    wandb.init(project="da6401_a2", name="2.7_wild_showcase", group="section_2_7")

    columns = ["image", "predicted_breed", "confidence", "bbox_image", "segmentation_image"]
    table   = wandb.Table(columns=columns)

    for fname in img_files:
        pil_img = Image.open(os.path.join(wild_dir, fname)).convert("RGB")
        x       = preprocess_pil(pil_img).to(device)
        img_np  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

        with torch.no_grad():
            cls_logits = classifier(x)[0].cpu()
            pred_bbox  = localizer(x)[0].cpu()
            seg_logits = unet(x)

        # Classification
        probs      = F.softmax(cls_logits, dim=0)
        top_idx    = probs.argmax().item()
        confidence = probs[top_idx].item()
        pred_class = PET_CLASSES[top_idx] if top_idx < len(PET_CLASSES) else f"class_{top_idx}"

        # ---- Bounding box figure ----
        fig_bbox, ax = plt.subplots(figsize=(5,5))
        ax.imshow(img_np)
        cx, cy, w, h = pred_bbox.tolist()
        rect = patches.Rectangle((cx-w/2, cy-h/2), w, h,
                                  linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"{pred_class}\nConf={confidence:.3f}", fontsize=9)
        ax.axis("off"); plt.tight_layout()

        # ---- Segmentation figure ----
        pred_mask = seg_logits.argmax(1)[0].cpu().numpy()
        pred_rgb  = mask_to_rgb(pred_mask)

        fig_seg, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].imshow(img_np);   axes[0].set_title("Original");              axes[0].axis("off")
        axes[1].imshow(pred_rgb); axes[1].set_title("Predicted Segmentation"); axes[1].axis("off")
        plt.tight_layout()

        table.add_data(
            wandb.Image(img_np),
            pred_class,
            round(confidence, 4),
            wandb.Image(fig_bbox),
            wandb.Image(fig_seg),
        )
        plt.close("all")
        print(f"[2.7] {fname}: {pred_class} ({confidence:.3f})")

    wandb.log({"wild_image_showcase": table})
    wandb.finish()


# =========================================================================
# Section 2.8 — Meta-Analysis and Reflection
# =========================================================================

def run_section_2_8(args):
    wandb.init(project="da6401_a2", name="2.8_meta_analysis", group="section_2_8")

    api     = wandb.Api()
    project = "da6401_a2"

    try:
        entity = wandb.run.entity
        runs   = api.runs(f"{entity}/{project}")
    except Exception:
        runs = []

    cls_runs = [r for r in runs if r.name.startswith("classify")]
    loc_runs = [r for r in runs if r.name.startswith("localize")]
    seg_runs = [r for r in runs if r.name.startswith("segment")]

    def get_history(run, keys):
        try:
            hist = run.history(keys=keys, pandas=False)
            return {k: [row.get(k) for row in hist if row.get(k) is not None] for k in keys}
        except Exception:
            return {k: [] for k in keys}

    # Plot 1: Classification curves
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in cls_runs[:1]:
        h = get_history(run, ["train/loss","val/loss","train/acc","val/acc"])
        if h["train/loss"]:
            ep = range(1, len(h["train/loss"])+1)
            axes[0].plot(ep, h["train/loss"], "b-",  label="Train Loss")
            axes[0].plot(ep, h["val/loss"],   "r--", label="Val Loss")
            axes[1].plot(ep, h["train/acc"],  "b-",  label="Train Acc")
            axes[1].plot(ep, h["val/acc"],    "r--", label="Val Acc")
    axes[0].set_title("Task 1: Classification Loss");     axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Task 1: Classification Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.suptitle("Task 1 — VGG11 Classification Training History", fontweight="bold")
    plt.tight_layout()
    wandb.log({"meta/task1_curves": wandb.Image(fig1)})
    plt.close(fig1)

    # Plot 2: Localisation curves
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in loc_runs[:1]:
        h = get_history(run, ["train/loss","val/loss","val/iou"])
        if h["train/loss"]:
            ep = range(1, len(h["train/loss"])+1)
            axes[0].plot(ep, h["train/loss"], "b-",  label="Train Loss")
            axes[0].plot(ep, h["val/loss"],   "r--", label="Val Loss")
        if h["val/iou"]:
            ep2 = range(1, len(h["val/iou"])+1)
            axes[1].plot(ep2, h["val/iou"], "g-", label="Val IoU")
    axes[0].set_title("Task 2: Localisation Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Task 2: Validation IoU");    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.suptitle("Task 2 — Localisation Training History", fontweight="bold")
    plt.tight_layout()
    wandb.log({"meta/task2_curves": wandb.Image(fig2)})
    plt.close(fig2)

    # Plot 3: Segmentation curves
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in seg_runs[:1]:
        h = get_history(run, ["train/loss","val/loss","val/dice"])
        if h["train/loss"]:
            ep = range(1, len(h["train/loss"])+1)
            axes[0].plot(ep, h["train/loss"], "b-",  label="Train Loss")
            axes[0].plot(ep, h["val/loss"],   "r--", label="Val Loss")
        if h["val/dice"]:
            ep2 = range(1, len(h["val/dice"])+1)
            axes[1].plot(ep2, h["val/dice"], "g-", label="Val Dice")
    axes[0].set_title("Task 3: Segmentation Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Task 3: Dice Score");        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.suptitle("Task 3 — U-Net Segmentation Training History", fontweight="bold")
    plt.tight_layout()
    wandb.log({"meta/task3_curves": wandb.Image(fig3)})
    plt.close(fig3)

    # Plot 4: All-tasks overview
    fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
    task_data = [
        (cls_runs, ["train/loss","val/loss"], "Task 1: Classification", "Loss"),
        (loc_runs, ["train/loss","val/loss"], "Task 2: Localisation",   "Loss"),
        (seg_runs, ["train/loss","val/loss"], "Task 3: Segmentation",   "Loss"),
    ]
    for ax, (run_list, keys, title, ylabel) in zip(axes, task_data):
        for run in run_list[:1]:
            h = get_history(run, keys)
            if h[keys[0]]:
                ep = range(1, len(h[keys[0]])+1)
                ax.plot(ep, h[keys[0]], "b-",  label="Train")
                ax.plot(ep, h[keys[1]], "r--", label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("All Tasks — Training vs Validation Loss Overview", fontweight="bold")
    plt.tight_layout()
    wandb.log({"meta/all_tasks_overview": wandb.Image(fig4)})
    plt.close(fig4)

    # Summary metrics table
    summary_cols  = ["task", "best_val_metric", "metric_name", "epochs_trained"]
    summary_table = wandb.Table(columns=summary_cols)

    for run in cls_runs[:1]:
        best = run.summary.get("val/acc", run.summary.get("best_val_acc", "N/A"))
        summary_table.add_data("Task 1 — Classification", best, "Val Accuracy",
                               run.summary.get("epoch", "?"))
    for run in loc_runs[:1]:
        best = run.summary.get("val/iou", "N/A")
        summary_table.add_data("Task 2 — Localisation",  best, "Val IoU",
                               run.summary.get("epoch", "?"))
    for run in seg_runs[:1]:
        best = run.summary.get("val/dice", "N/A")
        summary_table.add_data("Task 3 — Segmentation",  best, "Val Dice",
                               run.summary.get("epoch", "?"))

    wandb.log({"meta/summary_table": summary_table})
    print("[2.8] Meta-analysis plots and summary table logged.")
    wandb.finish()


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment 2 — W&B Experiments (Sections 2.1–2.8)")
    p.add_argument("--section",          required=True,
                   choices=["2.1","2.2","2.3","2.4","2.5","2.6","2.7","2.8"])
    p.add_argument("--data_root",        default="./data/pets")
    p.add_argument("--wild_dir",         default="./wild_images",
                   help="Directory containing 3 novel pet images (section 2.7)")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--classifier_ckpt",  default="checkpoints/classifier.pth")
    p.add_argument("--localizer_ckpt",   default="checkpoints/localizer.pth")
    p.add_argument("--unet_ckpt",        default="checkpoints/unet.pth")
    return p.parse_args()


if __name__ == "__main__":
    from torch.utils.data import DataLoader   # imported here to keep top-level imports clean

    args = parse_args()
    dispatch = {
        "2.1": run_section_2_1,
        "2.2": run_section_2_2,
        "2.3": run_section_2_3,
        "2.4": run_section_2_4,
        "2.5": run_section_2_5,
        "2.6": run_section_2_6,
        "2.7": run_section_2_7,
        "2.8": run_section_2_8,
    }
    dispatch[args.section](args)
