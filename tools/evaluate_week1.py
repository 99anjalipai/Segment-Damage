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

import matplotlib.pyplot as plt
import numpy as np

from data.cardd_dataset import build_dataloaders
from models import DamageSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 baseline evaluation.")
    parser.add_argument("--config", type=str, default="configs/week1_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/week1_unet/best.pt")
    parser.add_argument("--tiny-area-threshold", type=int, default=1500)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/week1_unet/eval",
        help="Directory to save evaluation metrics JSON."
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=5,
        help="Number of sample visualizations to save."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate: train, val, or test (if available)."
    )
    return parser.parse_args()
def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    img = image * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).detach().cpu().numpy()

@torch.no_grad()
def save_eval_visualizations(
    model: DamageSegmentor,
    dataset,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 5,
    ) -> None:
    model_was_training = model.training
    model.eval()

    # Randomly sample indices from the dataset
    import random
    num_samples = min(max_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    images = []
    masks = []
    preds = []
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"]
        with torch.no_grad():
            logits = model(image)["logits"]
            pred = logits.argmax(dim=1).squeeze(0).cpu()
        images.append(image.squeeze(0))
        masks.append(mask)
        preds.append(pred)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(num_samples):
        image_np = denormalize_image(images[i])
        gt_np = masks[i].detach().cpu().numpy()
        pred_np = preds[i].detach().cpu().numpy()
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=max(1, int(gt_np.max())))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_np, cmap="gray", vmin=0, vmax=max(1, int(pred_np.max())))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(vis_dir / f"eval_samples.png", dpi=150)
    plt.close(fig)
    if model_was_training:
        model.train()


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

    # Select split file based on args.split
    split_map = {
        "train": cfg["dataset"]["train_split"],
        "val": cfg["dataset"]["val_split"],
        "test": cfg["dataset"].get("test_split", cfg["dataset"].get("val_split")),
    }
    split_file = split_map[args.split]
    from data.cardd_dataset import CarDDSegmentationDataset, SegmentationPairTransform, TransformConfig
    transform = SegmentationPairTransform(TransformConfig(image_size=cfg["dataset"].get("image_size", 512)), is_train=False)
    dataset = CarDDSegmentationDataset(
        image_dir=cfg["dataset"]["image_dir"],
        mask_dir=cfg["dataset"]["mask_dir"],
        split_file=split_file,
        transform=transform,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
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
        loader=loader,
        device=device,
        tiny_area_threshold=args.tiny_area_threshold,
        num_classes=cfg["model"]["num_classes"],
    )
    print(json.dumps(metrics, indent=2))

    # Save metrics to results dir
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Info] Metrics saved to {metrics_path}")

    # Save visualizations
    print(f"[Stage] Saving evaluation visualizations to {results_dir / 'visualizations'} ...")
    save_eval_visualizations(model, dataset, device, results_dir, max_samples=args.visualize_samples)
    print(f"[Info] Visualization saved as {results_dir / 'visualizations' / 'eval_samples.png'}")


if __name__ == "__main__":
    main()
