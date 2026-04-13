"""
Oxford-IIIT Pet Dataset loader.

Provides three dataset variants:
  PetClassificationDataset  – image + class label (37 breeds, 0-indexed)
  PetLocalizationDataset    – image + bbox [cx, cy, w, h] in pixel coords (224-space)
  PetSegmentationDataset    – image + trimap mask long tensor (classes 0/1/2)

Expected directory layout after extraction:
  <root>/
    images/          <- *.jpg files
    annotations/
      trainval.txt   <- columns: ImageName ClassId Species BreedId
      test.txt
      xmls/          <- VOC-format bounding-box XML files
      trimaps/       <- *.png trimap masks  (1=fg, 2=bg, 3=boundary)

All images are normalised with ImageNet mean/std as the autograder test set
uses normalised images.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 224   # fixed per VGG11 paper (hardcoded as instructed)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_train_transforms() -> A.Compose:
    """Augmented transforms for classification / localisation training."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"],
                                 min_visibility=0.1),
    )


def get_val_transforms() -> A.Compose:
    """Deterministic transforms for classification / localisation validation."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )


def get_seg_train_transforms() -> A.Compose:
    """Augmented transforms for segmentation training (no bbox params)."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_seg_val_transforms() -> A.Compose:
    """Deterministic transforms for segmentation validation."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_split_file(list_file: str) -> List[Tuple[str, int]]:
    """
    Parse a trainval.txt / test.txt annotation list.

    Lines that start with '#' are comments. Each valid line has:
        <ImageName>  <ClassId>  <Species>  <BreedId>
    ClassId is 1-indexed -> converted to 0-indexed here.

    Returns list of (image_stem, class_id_0indexed).
    """
    entries: List[Tuple[str, int]] = []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            stem     = parts[0]
            class_id = int(parts[1]) - 1   # 1-indexed -> 0-indexed
            entries.append((stem, class_id))   # NOTE: tuple, not two args
    return entries


def _parse_voc_bbox(xml_path: str) -> Tuple[float, float, float, float]:
    """
    Parse a VOC-format XML file.

    Returns (x_min, y_min, box_w, box_h) in the *original* image pixel space
    as COCO format so albumentations can handle it directly.
    """
    tree   = ET.parse(xml_path)
    root   = tree.getroot()
    obj    = root.find("object")
    bndbox = obj.find("bndbox")
    xmin = float(bndbox.find("xmin").text)
    ymin = float(bndbox.find("ymin").text)
    xmax = float(bndbox.find("xmax").text)
    ymax = float(bndbox.find("ymax").text)
    return xmin, ymin, xmax - xmin, ymax - ymin


# ---------------------------------------------------------------------------
# Dataset: Classification
# ---------------------------------------------------------------------------

class PetClassificationDataset(Dataset):
    """
    Oxford-IIIT Pet – breed classification.

    Parameters
    ----------
    root      : str                     Dataset root directory.
    split     : str                     "trainval" or "test".
    transform : albumentations.Compose  Applied to each sample.
    """

    def __init__(self, root: str, split: str = "trainval", transform=None):
        self.root      = Path(root)
        self.transform = transform if transform is not None else get_val_transforms()

        list_file = self.root / "annotations" / f"{split}.txt"
        raw = _load_split_file(str(list_file))

        self.samples: List[Tuple[str, int]] = []
        for stem, class_id in raw:
            img_path = self.root / "images" / f"{stem}.jpg"
            if img_path.exists():
                self.samples.append((str(img_path), class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        out   = self.transform(image=image, bboxes=[], bbox_labels=[])
        return out["image"], torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset: Localisation
# ---------------------------------------------------------------------------

class PetLocalizationDataset(Dataset):
    """
    Oxford-IIIT Pet – single-object bounding-box regression.

    Returns
    -------
    image : FloatTensor (3, 224, 224)  normalised
    bbox  : FloatTensor (4,)  [x_center, y_center, width, height] in
            pixel coordinates of the 224x224 image.
    """

    def __init__(self, root: str, split: str = "trainval", transform=None):
        self.root      = Path(root)
        self.transform = transform if transform is not None else get_val_transforms()

        list_file = self.root / "annotations" / f"{split}.txt"
        raw       = _load_split_file(str(list_file))
        xml_dir   = self.root / "annotations" / "xmls"

        self.samples: List[Tuple[str, str]] = []
        for stem, _ in raw:
            img_path = self.root / "images" / f"{stem}.jpg"
            xml_path = xml_dir / f"{stem}.xml"
            if img_path.exists() and xml_path.exists():
                self.samples.append((str(img_path), str(xml_path)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        xmin, ymin, bw, bh = _parse_voc_bbox(xml_path)

        # Clamp to valid image bounds
        xmin = max(0.0, min(float(xmin), orig_w - 1))
        ymin = max(0.0, min(float(ymin), orig_h - 1))
        bw   = min(float(bw), orig_w - xmin)
        bh   = min(float(bh), orig_h - ymin)

        out = self.transform(
            image=image,
            bboxes=[[xmin, ymin, bw, bh]],
            bbox_labels=[0],
        )
        image_tensor = out["image"]

        if out["bboxes"]:
            ax, ay, aw, ah = out["bboxes"][0]
        else:
            # Fallback: manual scale if bbox was clipped away during augmentation
            sx = IMG_SIZE / orig_w
            sy = IMG_SIZE / orig_h
            ax, ay, aw, ah = xmin * sx, ymin * sy, bw * sx, bh * sy

        # COCO [x_min, y_min, w, h] -> [cx, cy, w, h]
        cx = ax + aw / 2.0
        cy = ay + ah / 2.0
        bbox = torch.tensor([cx, cy, float(aw), float(ah)], dtype=torch.float32)
        return image_tensor, bbox


# ---------------------------------------------------------------------------
# Dataset: Segmentation
# ---------------------------------------------------------------------------

class PetSegmentationDataset(Dataset):
    """
    Oxford-IIIT Pet – trimap semantic segmentation.

    Trimap PNG pixel values:
        1 = foreground pet body  ->  class 0
        2 = background           ->  class 1
        3 = boundary/uncertain   ->  class 2

    Returns
    -------
    image : FloatTensor (3, 224, 224)  normalised
    mask  : LongTensor  (224, 224)     values in {0, 1, 2}
    """

    def __init__(self, root: str, split: str = "trainval", transform=None):
        self.root      = Path(root)
        self.transform = transform if transform is not None else get_seg_val_transforms()

        list_file  = self.root / "annotations" / f"{split}.txt"
        raw        = _load_split_file(str(list_file))
        trimap_dir = self.root / "annotations" / "trimaps"

        self.samples: List[Tuple[str, str]] = []
        for stem, _ in raw:
            img_path  = self.root / "images" / f"{stem}.jpg"
            mask_path = trimap_dir / f"{stem}.png"
            if img_path.exists() and mask_path.exists():
                self.samples.append((str(img_path), str(mask_path)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))   # values 1,2,3

        # Remap: 1->0 (fg), 2->1 (bg), 3->2 (boundary)
        mask = (mask.astype(np.int64) - 1).clip(0, 2).astype(np.uint8)

        # Resize image (bilinear) and mask (nearest) to IMG_SIZE x IMG_SIZE
        image = np.array(Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE),
                                                        Image.BILINEAR))
        mask  = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE),
                                                       Image.NEAREST))

        # Apply normalisation + ToTensor to image
        out          = self.transform(image=image, bboxes=[], bbox_labels=[])
        image_tensor = out["image"]
        mask_tensor  = torch.from_numpy(mask.astype(np.int64)).long()

        return image_tensor, mask_tensor
