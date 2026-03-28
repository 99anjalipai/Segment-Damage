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
        class_labels_file: str | Path | None = None,
        default_class_label: int = 0,
        infer_class_from_mask: bool = False,
        multilabel_classification: bool = False,
        num_dent_classes: int | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.default_class_label = int(default_class_label)
        self.infer_class_from_mask = bool(infer_class_from_mask)
        self.multilabel_classification = bool(multilabel_classification)
        self.num_dent_classes = int(num_dent_classes) if num_dent_classes is not None else None
        self.class_labels: Dict[str, int | List[int]] = {}

        if class_labels_file:
            with open(class_labels_file, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
            if not isinstance(raw_labels, dict):
                raise ValueError("class_labels_file must contain a JSON object mapping sample id to class index.")

            labels_obj = raw_labels.get("labels") if isinstance(raw_labels.get("labels"), dict) else raw_labels
            parsed: Dict[str, int | List[int]] = {}
            for key, value in labels_obj.items():
                if isinstance(value, list):
                    parsed[str(key)] = [int(x) for x in value]
                else:
                    parsed[str(key)] = int(value)
            self.class_labels = parsed

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
        sample_id = row["image"].stem
        image = Image.open(row["image"]).convert("RGB")
        mask = Image.open(row["mask"]).convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)

        if sample_id in self.class_labels:
            raw_class_value = self.class_labels[sample_id]
            if isinstance(raw_class_value, list):
                class_indices = [int(v) for v in raw_class_value]
            else:
                class_indices = [int(raw_class_value)]
        elif self.infer_class_from_mask:
            fg_values = mask_tensor[mask_tensor > 0]
            class_indices = [int(fg_values.max().item())] if fg_values.numel() > 0 else [int(self.default_class_label)]
        else:
            class_indices = [int(self.default_class_label)]

        class_label = int(class_indices[0]) if class_indices else int(self.default_class_label)

        if self.multilabel_classification:
            if self.num_dent_classes is None or self.num_dent_classes <= 0:
                raise ValueError("num_dent_classes must be set when multilabel_classification is enabled.")
            class_label_multi = torch.zeros(int(self.num_dent_classes), dtype=torch.float32)
            for cls_idx in class_indices:
                if 0 <= int(cls_idx) < int(self.num_dent_classes):
                    class_label_multi[int(cls_idx)] = 1.0
        else:
            class_label_multi = torch.zeros(1, dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "class_label_multi": class_label_multi,
            "id": sample_id,
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
    class_labels_file: str | Path | None = None,
    default_class_label: int = 0,
    infer_class_from_mask: bool = False,
    multilabel_classification: bool = False,
    num_dent_classes: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    train_transform = SegmentationPairTransform(TransformConfig(image_size=image_size), is_train=True)
    val_transform = SegmentationPairTransform(TransformConfig(image_size=image_size), is_train=False)

    train_dataset = CarDDSegmentationDataset(
        image_dir,
        mask_dir,
        train_split,
        train_transform,
        class_labels_file=class_labels_file,
        default_class_label=default_class_label,
        infer_class_from_mask=infer_class_from_mask,
        multilabel_classification=multilabel_classification,
        num_dent_classes=num_dent_classes,
    )
    val_dataset = CarDDSegmentationDataset(
        image_dir,
        mask_dir,
        val_split,
        val_transform,
        class_labels_file=class_labels_file,
        default_class_label=default_class_label,
        infer_class_from_mask=infer_class_from_mask,
        multilabel_classification=multilabel_classification,
        num_dent_classes=num_dent_classes,
    )

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
