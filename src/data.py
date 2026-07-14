"""Dataset loading, stratified splitting and augmentation.

Class convention throughout the project:
    0 = ok_front   (good part)
    1 = def_front  (defective part)   <-- the POSITIVE class

Class 1 is the positive class on purpose: in inspection we care about
detecting defects, so "recall" always means "recall of defects".
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet statistics — required when using pretrained backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["ok", "defect"]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Augmentation policy.

    Note what is NOT here: no vertical flip beyond what is physically
    plausible, no aggressive colour jitter. The impeller images are captured
    under fixed factory lighting, so augmenting colour would simulate a
    condition that never occurs in deployment and would only add noise.
    Rotation and small translations DO occur (part placement varies), so those
    are included.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _stratified_split(targets: list[int], val_split: float, seed: int) -> Tuple[list[int], list[int]]:
    """Split indices per class so the val set keeps the original class balance."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    targets_arr = np.asarray(targets)
    for cls in np.unique(targets_arr):
        cls_idx = np.flatnonzero(targets_arr == cls)
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_split))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Return train/val/test loaders plus class weights for the loss.

    The Kaggle casting dataset ships with train/ and test/ directories. We do
    NOT touch test/ during development — the validation set is carved out of
    train/. Tuning anything on the test set would inflate the reported numbers
    and is the single most common mistake in student portfolio projects.
    """
    root = Path(cfg["data"]["root"])
    train_dir, test_dir = root / "train", root / "test"

    for d in (train_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Expected directory not found: {d}\n"
                "Download the dataset (see README) and check data.root in configs/config.yaml."
            )

    size = cfg["data"]["image_size"]
    seed = cfg["data"]["seed"]

    # Two views of the same folder: one augmented (train), one clean (val).
    full_train_aug = datasets.ImageFolder(train_dir, build_transforms(size, train=True))
    full_train_clean = datasets.ImageFolder(train_dir, build_transforms(size, train=False))
    test_set = datasets.ImageFolder(test_dir, build_transforms(size, train=False))

    # ImageFolder sorts alphabetically: def_front -> 0, ok_front -> 1.
    # We want defect = 1, so we detect and remap rather than assume.
    defect_key = [c for c in full_train_aug.classes if c.startswith("def")][0]
    defect_original_idx = full_train_aug.class_to_idx[defect_key]
    remap_needed = defect_original_idx != 1

    train_idx, val_idx = _stratified_split(
        full_train_aug.targets, cfg["data"]["val_split"], seed
    )

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_clean, val_idx)

    if remap_needed:
        train_set = _RemapLabels(train_set)
        val_set = _RemapLabels(val_set)
        test_set = _RemapLabels(test_set)

    # Class weights, computed on the training split only.
    train_targets = np.asarray(full_train_aug.targets)[train_idx]
    if remap_needed:
        train_targets = 1 - train_targets
    counts = np.bincount(train_targets, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    common = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    print(f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    print(f"train class counts (ok, defect) = {counts.tolist()}")
    print(f"class weights (ok, defect)      = {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_weights


class _RemapLabels(torch.utils.data.Dataset):
    """Flip binary labels so that defect == 1."""

    def __init__(self, base: torch.utils.data.Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, 1 - y
