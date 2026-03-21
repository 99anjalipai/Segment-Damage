from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
MASK_SUFFIXES = (".png", ".bmp", ".tif", ".tiff")


@dataclass
class TransformConfig:
    image_size: int = 512
    horizontal_flip_prob: float = 0.5


class SegmentationPairTransform:
    def __init__(self, config: TransformConfig, is_train: bool) -> None:
        self.config = config
        self.is_train = is_train

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = F.resize(image, [self.config.image_size, self.config.image_size], InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.config.image_size, self.config.image_size], InterpolationMode.NEAREST)

        if self.is_train and random.random() < self.config.horizontal_flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_np = np.array(mask, dtype=np.int64)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        # CarDD SOD masks are commonly encoded as 0/255; convert to class ids 0/1.
        unique_vals = np.unique(mask_np)
        if unique_vals.size <= 2 and 0 in unique_vals and np.any(unique_vals > 1):
            mask_np = (mask_np > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


class CarDDSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        split_file: str | Path,
        transform: Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        with open(split_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        self.items = self._resolve_pairs(split_info["items"])
        if not self.items:
            raise ValueError("No valid image-mask pairs found for this split.")

    def _resolve_pairs(self, stems: List[str]) -> List[Dict[str, Path]]:
        pairs: List[Dict[str, Path]] = []
        for stem in stems:
            image_path = self._find_path(self.image_dir, stem, IMAGE_SUFFIXES)
            mask_path = self._find_path(self.mask_dir, stem, MASK_SUFFIXES)
            if image_path is None or mask_path is None:
                continue
            pairs.append({"image": image_path, "mask": mask_path})
        return pairs

    @staticmethod
    def _find_path(base_dir: Path, stem: str, suffixes: Tuple[str, ...]) -> Path | None:
        for suffix in suffixes:
            candidate = base_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.items[index]
        image = Image.open(row["image"]).convert("RGB")
        mask = Image.open(row["mask"]).convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": row["image"].stem,
        }


def build_dataloaders(
    image_dir: str | Path,
    mask_dir: str | Path,
    train_split: str | Path,
    val_split: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_transform = SegmentationPairTransform(TransformConfig(image_size=image_size), is_train=True)
    val_transform = SegmentationPairTransform(TransformConfig(image_size=image_size), is_train=False)

    train_dataset = CarDDSegmentationDataset(image_dir, mask_dir, train_split, train_transform)
    val_dataset = CarDDSegmentationDataset(image_dir, mask_dir, val_split, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
