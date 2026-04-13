# DA6401 Assignment 2 — Multi-Task Learning on Oxford-IIIT Pets

A PyTorch implementation of a **multi-task deep learning pipeline** built on a VGG11 backbone, trained on the Oxford-IIIT Pets dataset to jointly perform:

1. **Task 1 — Classification**: Predict the breed of the pet (37 classes)
2. **Task 2 — Localization**: Predict bounding boxes around the pet (IoU loss)
3. **Task 3 — Segmentation**: Generate pixel-level segmentation masks (Dice/cross-entropy loss)

Experiment tracking is done via **Weights & Biases (WandB)**.

---

Github--```https://github.com/suryapratap2002/DA6401_Assignment_2_AM24M015 ```

Wandb Report-- ```https://api-jungle.wandb.ai/links/spsinghiitian2020-iitmaana/2vi2w6on```

## Project Structure

```
da6401_assignment_2/
├── data/
│   ├── pets_dataset.py        # Dataset class for Oxford-IIIT Pets
│   └── __init__.py
├── models/
│   ├── vgg11.py               # VGG11 feature extractor (backbone)
│   ├── layers.py              # Custom layer definitions
│   ├── localization.py        # Localization head (bounding box regression)
│   ├── segmentation.py        # Segmentation head (pixel-wise prediction)
│   ├── multitask.py           # Unified multi-task model
│   └── __init__.py
├── losses/
│   ├── iou_loss.py            # IoU loss for bounding box regression
│   ├── seg_loss.py            # Segmentation loss (Dice + cross-entropy)
│   └── __init__.py
├── checkpoints/
│   └── checkpoints.md         # Notes on saved model checkpoints
├── wild_images/
│   ├── first.jpg              # Test images for inference on unseen data
│   ├── second.jpg
│   └── third.jpg
├── train.py                   # Main training script
├── inference.py               # Inference script for wild images
├── wandb_all_sections.py      # WandB logging utilities (all tasks)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key libraries used:

- `torch` & `torchvision` — model building and training
- `wandb` — experiment tracking and visualization
- `Pillow` — image loading and preprocessing

---

## Dataset

This project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which provides:

- 37 pet breeds (cats & dogs) with ~200 images per class
- Bounding box annotations
- Pixel-level trimap segmentation masks

The dataset is handled by `data/pets_dataset.py`, which returns images along with all three label types (class, bounding box, segmentation mask) for multi-task training.

---

## Training

```bash
python train.py
```

Training logs metrics to WandB automatically. You can configure hyperparameters (learning rate, batch size, number of epochs, etc.) directly in `train.py` or via command-line arguments.

To track your runs online, log in to WandB first:

```bash
wandb login
```

---

## Inference

To run inference on custom images:

```bash
python inference.py
```

Wild test images are stored in `wild_images/`. The inference script visualises all three task outputs (predicted class, bounding box, and segmentation mask) for each image.

---

## Model Architecture

The backbone is a **VGG11** convolutional network used as a shared feature extractor. Three task-specific heads are attached:

| Task | Head | Loss |
|---|---|---|
| Classification | Fully-connected classifier | Cross-entropy |
| Localization | Bounding-box regression head | IoU Loss |
| Segmentation | Upsampling decoder | Dice + Pixel-wise CE |

The combined loss is a weighted sum across all three task losses, optimised jointly.

---

## Experiment Tracking (WandB)

All training runs are logged to WandB. Tracked metrics include:

- Per-task train/validation loss curves
- Classification accuracy
- IoU score (localization)
- Dice score & pixel accuracy (segmentation)
- Feature map visualisations (first & last conv layers)
- Activation distribution histograms per epoch
- Segmentation sample overlays
- Wild image showcase

The `wandb_all_sections.py` script handles consolidated multi-task logging.

---

## Results

Training and evaluation results are available in the linked WandB project. Key logged artifacts include:

- `task1_curves` — classification training curves
- `task2_curves` — localization training curves
- `task3_curves` — segmentation training curves
- `all_tasks_overview` — unified summary panel
- `detection_results` — bounding box predictions table
- `segmentation_samples` — mask prediction overlays
- `wild_image_showcase` — qualitative results on held-out images

---

## Repository Notes

> **Note:** The `wandb/` directory contains local run artefacts and logs. It is recommended to add `wandb/` to `.gitignore` before pushing to GitHub to keep the repository lightweight.


---

## Course Information

| Field | Details |
|---|---|
| Course | DA6401 — Deep Learning |
| Assignment | Assignment 2 |
| Dataset | Oxford-IIIT Pets |
| Framework | PyTorch |
| Experiment Tracking | Weights & Biases |
