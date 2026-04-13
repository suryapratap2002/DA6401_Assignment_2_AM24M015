"""
# Run inference on a single image
python inference.py --image path/to/image.jpg

# Evaluate on test set
python inference.py --eval --data_root ./data/pets

# Quick sanity check
python inference.py --sanity
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel

# ImageNet normalisation (must match training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 224

PET_CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]


def preprocess_image(img_path: str) -> torch.Tensor:
    image = np.array(Image.open(img_path).convert("RGB"))
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"]
    return tensor.unsqueeze(0)


def run_single_image(model: MultiTaskPerceptionModel, img_path: str, device: torch.device):
    model.eval()
    x = preprocess_image(img_path).to(device)
    with torch.no_grad():
        cls_logits, bbox, seg_logits = model(x)

    cls_probs  = F.softmax(cls_logits, dim=1)[0]
    top5_probs, top5_idx = cls_probs.topk(5)
    seg_mask   = seg_logits.argmax(dim=1)[0].cpu().numpy()

    print("\n=== Classification (Top 5) ===")
    for i in range(5):
        cls_name = PET_CLASSES[top5_idx[i].item()] if top5_idx[i] < len(PET_CLASSES) else f"class_{top5_idx[i]}"
        print(f"  {cls_name:<40s}  {top5_probs[i].item():.4f}")

    cx, cy, w, h = bbox[0].cpu().tolist()
    print(f"\n=== Bounding Box (pixels, 224×224) ===")
    print(f"  cx={cx:.1f}  cy={cy:.1f}  w={w:.1f}  h={h:.1f}")
    print(f"  → x1={cx-w/2:.1f}  y1={cy-h/2:.1f}  x2={cx+w/2:.1f}  y2={cy+h/2:.1f}")

    unique, counts = np.unique(seg_mask, return_counts=True)
    print(f"\n=== Segmentation mask (224×224) ===")
    class_names = {0: "foreground", 1: "background", 2: "boundary"}
    for u, c in zip(unique, counts):
        print(f"  class {u} ({class_names.get(u, '?')}): {c} px  ({100*c/seg_mask.size:.1f}%)")
    return cls_logits, bbox, seg_logits


def evaluate_test_set(model: MultiTaskPerceptionModel, data_root: str, device: torch.device):
    from data.pets_dataset import PetClassificationDataset, get_val_transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    ds = PetClassificationDataset(data_root, split="test", transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            cls_logits, _, _ = model(imgs)
            preds = cls_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\nClassification Macro-F1 on test set: {f1:.4f}")
    return f1


def sanity_check():
    print("Running sanity check (no data needed)…")
    model = MultiTaskPerceptionModel(device="cpu")
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cls, bbox, seg = model(dummy)
    assert cls.shape  == (2, 37),        f"cls shape wrong: {cls.shape}"
    assert bbox.shape == (2, 4),         f"bbox shape wrong: {bbox.shape}"
    assert seg.shape  == (2, 3, 224, 224), f"seg shape wrong: {seg.shape}"
    print("  ✓ cls_logits :", cls.shape)
    print("  ✓ bbox       :", bbox.shape)
    print("  ✓ seg_logits :", seg.shape)
    print("Sanity check PASSED.")


def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 Inference")
    p.add_argument("--image",     default=None,          help="Path to a single image")
    p.add_argument("--eval",      action="store_true",   help="Run eval on test set")
    p.add_argument("--sanity",    action="store_true",   help="Run a quick sanity check")
    p.add_argument("--data_root", default="./data/pets", help="Dataset root for --eval")
    p.add_argument("--device",    default="cpu",         help="Device: cpu or cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sanity:
        sanity_check()
    else:
        device = torch.device(args.device)
        model  = MultiTaskPerceptionModel(device=args.device)
        model.eval()

        if args.image:
            run_single_image(model, args.image, device)

        if args.eval:
            evaluate_test_set(model, args.data_root, device)
