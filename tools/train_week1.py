from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.cardd_dataset import build_dataloaders
from models import DamageSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 baseline training for damage segmentation.")
    parser.add_argument("--config", type=str, default="configs/week1_unet.yaml")
    return parser.parse_args()


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    img = image * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).detach().cpu().numpy()


@torch.no_grad()
def save_epoch_visualization(
    model: DamageSegmentor,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_samples: int = 3,
) -> None:
    model_was_training = model.training
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    logits = model(images)["logits"]
    preds = logits.argmax(dim=1)

    num_samples = min(max_samples, images.shape[0])
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
    fig.savefig(vis_dir / f"epoch_{epoch:03d}.png", dpi=150)
    plt.close(fig)

    if model_was_training:
        model.train()


@torch.no_grad()
def evaluate(model: DamageSegmentor, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    running = {
        "loss_total": 0.0,
        "loss_ce": 0.0,
        "loss_dice": 0.0,
        "loss_focal": 0.0,
        "loss_grad": 0.0,
    }
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        losses = model.compute_losses(outputs["logits"], masks)
        for k in running:
            running[k] += losses[k].item()
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        "val_loss": running["loss_total"] / denom,
        "val_loss_ce": running["loss_ce"] / denom,
        "val_loss_dice": running["loss_dice"] / denom,
        "val_loss_focal": running["loss_focal"] / denom,
        "val_loss_grad": running["loss_grad"] / denom,
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: Dict,
    epochs: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None, str | None]:
    scheduler_cfg = training_cfg.get("scheduler")
    if not scheduler_cfg:
        return None, None

    name = str(scheduler_cfg.get("name", "")).strip().lower()
    if not name:
        return None, None

    if name in {"cosine", "cosine_annealing", "cosineannealing", "cosineannealinglr"}:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", epochs)),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
        return scheduler, "epoch"

    if name in {"reduce_on_plateau", "reducelronplateau", "lronplateau", "plateau"}:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 5)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
            cooldown=int(scheduler_cfg.get("cooldown", 0)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )
        return scheduler, "val_loss"

    raise ValueError(
        f"Unsupported scheduler '{name}'. Use one of: cosine_annealing, reduce_on_plateau"
    )


def build_optimizer(model: DamageSegmentor, training_cfg: Dict) -> torch.optim.Optimizer:
    optimizer_cfg = training_cfg.get("optimizer", {})
    name = str(optimizer_cfg.get("name", "adamw")).strip().lower()
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(optimizer_cfg.get("momentum", 0.9)),
            nesterov=bool(optimizer_cfg.get("nesterov", False)),
        )

    if name == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(optimizer_cfg.get("momentum", 0.0)),
            alpha=float(optimizer_cfg.get("alpha", 0.99)),
        )

    raise ValueError(f"Unsupported optimizer '{name}'. Use one of: adamw, adam, sgd, rmsprop")


def train() -> None:
    print("[Stage] Parsing configuration...")
    args = parse_args()
    cfg = load_config(args.config)

    print("[Stage] Selecting device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Info] Using device: {device}")

    print("[Stage] Data loading...")
    train_loader, val_loader = build_dataloaders(
        image_dir=cfg["dataset"]["image_dir"],
        mask_dir=cfg["dataset"]["mask_dir"],
        train_split=cfg["dataset"]["train_split"],
        val_split=cfg["dataset"]["val_split"],
        image_size=cfg["dataset"].get("image_size", 512),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    print(f"[Info] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print("[Stage] Model loading...")
    model = DamageSegmentor(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"].get("in_channels", 3),
        base_channels=cfg["model"].get("base_channels", 32),
        loss_config=cfg["training"].get("loss", {}),
    ).to(device)

    optimizer = build_optimizer(model=model, training_cfg=cfg["training"])
    scheduler, scheduler_step_on = build_scheduler(
        optimizer=optimizer,
        training_cfg=cfg["training"],
        epochs=cfg["training"].get("epochs", 25),
    )

    output_dir = Path(cfg["training"].get("output_dir", "outputs/week1_unet"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_enabled = cfg["training"].get("tensorboard", True)
    tb_dir = Path(cfg["training"].get("tensorboard_dir", output_dir / "tensorboard"))
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_enabled else None

    best_val_loss = float("inf")
    history = []
    vis_every = cfg["training"].get("visualize_every", 10)
    vis_samples = cfg["training"].get("visualize_samples", 3)

    if writer is not None:
        print(f"[Info] TensorBoard logging enabled at: {tb_dir}")
    if scheduler is not None:
        print(
            f"[Info] Scheduler enabled: {cfg['training']['scheduler']['name']} "
            f"(step on: {scheduler_step_on})"
        )

    epochs = cfg["training"].get("epochs", 25)
    print("[Stage] Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_dice": 0.0,
            "loss_focal": 0.0,
            "loss_grad": 0.0,
        }
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
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
            progress.set_postfix(
                loss=f"{losses['loss_total'].item():.4f}",
                ce=f"{losses['loss_ce'].item():.4f}",
                dice=f"{losses['loss_dice'].item():.4f}",
                focal=f"{losses['loss_focal'].item():.4f}",
                grad=f"{losses['loss_grad'].item():.4f}",
            )

        train_metrics = {k: v / max(num_batches, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)
        print(row)

        if scheduler is not None:
            if scheduler_step_on == "val_loss":
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()

        if writer is not None:
            writer.add_scalar("train/loss_total", train_metrics["loss_total"], epoch)
            writer.add_scalar("train/loss_ce", train_metrics["loss_ce"], epoch)
            writer.add_scalar("train/loss_dice", train_metrics["loss_dice"], epoch)
            if "loss_focal" in train_metrics:
                writer.add_scalar("train/loss_focal", train_metrics["loss_focal"], epoch)
            if "loss_grad" in train_metrics:
                writer.add_scalar("train/loss_grad", train_metrics["loss_grad"], epoch)
            writer.add_scalar("val/loss_total", val_metrics["val_loss"], epoch)
            writer.add_scalar("val/loss_ce", val_metrics["val_loss_ce"], epoch)
            writer.add_scalar("val/loss_dice", val_metrics["val_loss_dice"], epoch)
            writer.add_scalar("val/loss_focal", val_metrics["val_loss_focal"], epoch)
            writer.add_scalar("val/loss_grad", val_metrics["val_loss_grad"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if vis_every > 0 and epoch % vis_every == 0:
            print(f"[Stage] Saving visualizations for epoch {epoch}...")
            save_epoch_visualization(model, val_loader, device, output_dir, epoch, max_samples=vis_samples)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scheduler_name": cfg["training"].get("scheduler", {}).get("name"),
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(checkpoint, output_dir / "best.pt")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    train()
