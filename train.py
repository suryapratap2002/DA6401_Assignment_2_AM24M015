"""
-----
# Task 1 – VGG11 Classification
python train.py --task classify --data_root ./data/pets --epochs 30 --lr 1e-3

# Task 2 – Localization
python train.py --task localize --data_root ./data/pets --epochs 30 --lr 5e-4  --classifier_ckpt checkpoints/classifier.pth

# Task 3 – Segmentation
python train.py --task segment --data_root ./data/pets --epochs 40 --lr 1e-4 --classifier_ckpt checkpoints/classifier.pth
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import UNet
from losses.iou_loss import IoULoss
from losses.seg_loss import CombinedSegLoss
from data.pets_dataset import (
    PetClassificationDataset,
    PetLocalizationDataset,
    PetSegmentationDataset,
    get_train_transforms,
    get_val_transforms,
    get_seg_train_transforms,
    get_seg_val_transforms,
)

def save_checkpoint(model: nn.Module, path: str, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  Saved → {path}")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_dice(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3, smooth: float = 1.0) -> float:
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
    p_flat = probs.view(probs.shape[0], num_classes, -1)
    t_flat = targets_oh.view(targets_oh.shape[0], num_classes, -1)
    inter = (p_flat * t_flat).sum(2)
    card  = p_flat.sum(2) + t_flat.sum(2)
    dice  = (2 * inter + smooth) / (card + smooth)
    return dice.mean().item()


# =========================================================================
# Task 1: Classification
# =========================================================================

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classify] Device: {device}")

    train_ds = PetClassificationDataset(args.data_root, split="trainval",
                                        transform=get_train_transforms())
    val_ds   = PetClassificationDataset(args.data_root, split="test",
                                        transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)

    model = VGG11(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project="da6401_a2", name="classify", config=vars(args))

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc  += compute_accuracy(logits, labels)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                val_acc  += compute_accuracy(logits, labels)
        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)

        scheduler.step()

        wandb.log({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc,
                   "val/loss": val_loss, "val/acc": val_acc})
        print(f"Ep {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, "checkpoints/classifier.pth",
                            {"epoch": epoch, "val_acc": val_acc})

    wandb.finish()
    print(f"[Classify] Best val acc: {best_val_acc:.4f}")


# Task 2: Localization

def train_localizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Localize] Device: {device}")

    train_ds = PetLocalizationDataset(args.data_root, split="trainval",
                                      transform=get_train_transforms())
    val_ds   = PetLocalizationDataset(args.data_root, split="test",
                                      transform=get_val_transforms())

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val   samples: {len(val_ds)}")

    # If test split has no XML annotations fall back to a 80/20 split of trainval
    if len(val_ds) == 0:
        print("  WARNING: test split has no bbox annotations.")
        print("  Falling back to 80/20 split of trainval for validation.")
        from torch.utils.data import random_split
        full_ds = PetLocalizationDataset(args.data_root, split="trainval",
                                         transform=get_val_transforms())
        n_val   = max(1, int(0.2 * len(full_ds)))
        n_train = len(full_ds) - n_val
        # Re-build train_ds with augmentation transforms
        train_full = PetLocalizationDataset(args.data_root, split="trainval",
                                            transform=get_train_transforms())
        indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42)).tolist()
        train_ds = torch.utils.data.Subset(train_full, indices[:n_train])
        val_ds   = torch.utils.data.Subset(full_ds,   indices[n_train:])
        print(f"  Adjusted  Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)

    # Load pre-trained backbone
    vgg = VGG11(num_classes=37)
    if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
        state = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        vgg.load_state_dict(state, strict=False)
        print(f"  Loaded classifier backbone from {args.classifier_ckpt}")

    model = LocalizationModel(vgg=vgg, freeze_backbone=args.freeze_backbone).to(device)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    wandb.init(project="da6401_a2", name="localize", config=vars(args))

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        t_loss = 0.0
        for imgs, bboxes in train_loader:
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        v_loss, v_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, bboxes in val_loader:
                imgs, bboxes = imgs.to(device), bboxes.to(device)
                preds = model(imgs)
                v_loss += (mse_loss(preds, bboxes) + iou_loss(preds, bboxes)).item()
                # Mean IoU for logging
                iou_vals = 1.0 - iou_loss(preds, bboxes, reduction="none")
                v_iou += iou_vals.mean().item()
        n_val_batches = len(val_loader)
        v_loss = v_loss / n_val_batches if n_val_batches > 0 else 0.0
        v_iou  = v_iou  / n_val_batches if n_val_batches > 0 else 0.0

        scheduler.step()

        wandb.log({"epoch": epoch, "train/loss": t_loss, "val/loss": v_loss, "val/iou": v_iou})
        print(f"Ep {epoch:3d} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} iou={v_iou:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(model, "checkpoints/localizer.pth",
                            {"epoch": epoch, "val_loss": v_loss})

    wandb.finish()


# Task 3: Segmentation

def train_segmenter(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Segment] Device: {device}")

    train_ds = PetSegmentationDataset(args.data_root, split="trainval",
                                      transform=get_seg_train_transforms())
    val_ds   = PetSegmentationDataset(args.data_root, split="test",
                                      transform=get_seg_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)

    # Load pre-trained backbone
    vgg = VGG11(num_classes=37)
    if args.classifier_ckpt and os.path.isfile(args.classifier_ckpt):
        state = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        vgg.load_state_dict(state, strict=False)
        print(f"  Loaded classifier backbone from {args.classifier_ckpt}")

    model = UNet(vgg=vgg, num_classes=3, freeze_encoder=args.freeze_backbone).to(device)

    criterion = CombinedSegLoss(dice_weight=0.5, ce_weight=0.5)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project="da6401_a2", name="segment", config=vars(args))

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        t_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        v_loss, v_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                v_loss += criterion(logits, masks).item()
                v_dice += compute_dice(logits, masks)
        v_loss /= len(val_loader)
        v_dice /= len(val_loader)

        scheduler.step()

        wandb.log({"epoch": epoch, "train/loss": t_loss, "val/loss": v_loss, "val/dice": v_dice})
        print(f"Ep {epoch:3d} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} dice={v_dice:.4f}")

        if v_dice > best_dice:
            best_dice = v_dice
            save_checkpoint(model, "checkpoints/unet.pth",
                            {"epoch": epoch, "val_dice": v_dice})

    wandb.finish()

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 Training")
    p.add_argument("--task", required=True, choices=["classify", "localize", "segment"],
                   help="Which task to train")
    p.add_argument("--data_root", default="./data/pets",
                   help="Root directory of the Oxford-IIIT Pet dataset")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout_p",  type=float, default=0.5,
                   help="CustomDropout probability for classifier")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze VGG11 backbone when training localizer/segmenter")
    p.add_argument("--classifier_ckpt", default="checkpoints/classifier.pth",
                   help="Path to pre-trained classifier checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "classify":
        train_classifier(args)
    elif args.task == "localize":
        train_localizer(args)
    elif args.task == "segment":
        train_segmenter(args)