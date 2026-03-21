from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.cardd_dataset import build_dataloaders
from models import DamageSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 baseline training for damage segmentation.")
    parser.add_argument("--config", type=str, default="configs/week1_unet.yaml")
    return parser.parse_args()


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model: DamageSegmentor, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        losses = model.compute_losses(outputs["logits"], masks)
        total_loss += losses["loss_total"].item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return {"val_loss": avg_loss}


def train() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(
        image_dir=cfg["dataset"]["image_dir"],
        mask_dir=cfg["dataset"]["mask_dir"],
        train_split=cfg["dataset"]["train_split"],
        val_split=cfg["dataset"]["val_split"],
        image_size=cfg["dataset"].get("image_size", 512),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
    )

    model = DamageSegmentor(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"].get("in_channels", 3),
        base_channels=cfg["model"].get("base_channels", 32),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"].get("lr", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    output_dir = Path(cfg["training"].get("output_dir", "outputs/week1_unet"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    epochs = cfg["training"].get("epochs", 25)
    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss_total": 0.0, "loss_ce": 0.0, "loss_dice": 0.0}
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = model.compute_losses(outputs["logits"], masks)
            losses["loss_total"].backward()
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            num_batches += 1

        train_metrics = {k: v / max(num_batches, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)
        print(row)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(checkpoint, output_dir / "best.pt")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train()
