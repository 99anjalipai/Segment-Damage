from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from data.cardd_dataset import build_dataloaders
from models import DamageSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 baseline evaluation.")
    parser.add_argument("--config", type=str, default="configs/week1_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/week1_unet/best.pt")
    parser.add_argument("--tiny-area-threshold", type=int, default=1500)
    return parser.parse_args()


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: DamageSegmentor,
    loader: DataLoader,
    device: torch.device,
    tiny_area_threshold: int,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    iou_numer = torch.zeros(num_classes, dtype=torch.float64)
    iou_denom = torch.zeros(num_classes, dtype=torch.float64)

    tiny_tp = 0
    tiny_fn = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)["logits"]
        preds = logits.argmax(dim=1)

        for cls in range(num_classes):
            pred_cls = preds == cls
            mask_cls = masks == cls
            intersection = (pred_cls & mask_cls).sum().item()
            union = (pred_cls | mask_cls).sum().item()
            iou_numer[cls] += intersection
            iou_denom[cls] += union

        # Tiny-damage DET_l proxy: recall over tiny foreground connected area criteria.
        foreground = masks > 0
        tiny_items = foreground.flatten(1).sum(dim=1) <= tiny_area_threshold
        if tiny_items.any():
            fg_pred = preds > 0
            fg_true = foreground
            for i in torch.where(tiny_items)[0].tolist():
                has_true = fg_true[i].any().item()
                has_hit = (fg_pred[i] & fg_true[i]).any().item()
                if has_true and has_hit:
                    tiny_tp += 1
                elif has_true:
                    tiny_fn += 1

    iou_per_class = (iou_numer / torch.clamp(iou_denom, min=1.0)).tolist()
    miou = float(sum(iou_per_class) / len(iou_per_class))
    det_l = float(tiny_tp / max(tiny_tp + tiny_fn, 1))
    f1_from_miou = float((2 * miou) / max(miou + 1.0, 1e-8))

    return {
        "mIoU": miou,
        "IoU_per_class": iou_per_class,
        "F1_proxy": f1_from_miou,
        "DET_l": det_l,
        "tiny_true_positive": tiny_tp,
        "tiny_false_negative": tiny_fn,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, val_loader = build_dataloaders(
        image_dir=cfg["dataset"]["image_dir"],
        mask_dir=cfg["dataset"]["mask_dir"],
        train_split=cfg["dataset"]["train_split"],
        val_split=cfg["dataset"]["val_split"],
        image_size=cfg["dataset"].get("image_size", 512),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    model = DamageSegmentor(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"].get("in_channels", 3),
        base_channels=cfg["model"].get("base_channels", 32),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        tiny_area_threshold=args.tiny_area_threshold,
        num_classes=cfg["model"]["num_classes"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
