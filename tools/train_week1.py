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


def classification_metrics_from_confusion(conf: torch.Tensor) -> Dict[str, float]:
    total = float(conf.sum().item())
    if total <= 0.0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    correct = float(torch.diag(conf).sum().item())
    accuracy = correct / total

    f1_scores = []
    num_classes = int(conf.shape[0])
    for cls_idx in range(num_classes):
        tp = float(conf[cls_idx, cls_idx].item())
        fp = float(conf[:, cls_idx].sum().item() - tp)
        fn = float(conf[cls_idx, :].sum().item() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        if precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    macro_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    return {"accuracy": float(accuracy), "macro_f1": macro_f1}


def multilabel_metrics_from_counts(
    tp_per_class: torch.Tensor,
    fp_per_class: torch.Tensor,
    fn_per_class: torch.Tensor,
    exact_match_correct: int,
    num_samples: int,
) -> Dict[str, float]:
    if num_samples <= 0:
        return {
            "exact_match": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
        }

    tp = float(tp_per_class.sum().item())
    fp = float(fp_per_class.sum().item())
    fn = float(fn_per_class.sum().item())

    micro_precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    if micro_precision + micro_recall > 0.0:
        micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    f1_scores = []
    for cls_idx in range(int(tp_per_class.shape[0])):
        tp_c = float(tp_per_class[cls_idx].item())
        fp_c = float(fp_per_class[cls_idx].item())
        fn_c = float(fn_per_class[cls_idx].item())
        precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0.0 else 0.0
        recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0.0 else 0.0
        if precision_c + recall_c > 0.0:
            f1_c = 2.0 * precision_c * recall_c / (precision_c + recall_c)
        else:
            f1_c = 0.0
        f1_scores.append(f1_c)

    macro_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    exact_match = float(exact_match_correct / max(num_samples, 1))
    return {
        "exact_match": exact_match,
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 baseline training for damage segmentation.")
    parser.add_argument("--config", type=str, default="configs/mask2former_tiny.yaml")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from the config (e.g. 1 for smoke tests).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load best.pt from output_dir (start from pretrained weights).",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop each training epoch after this many batches (smoke tests; omit for full epoch).",
    )
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
def evaluate(
    model: DamageSegmentor,
    loader: DataLoader,
    device: torch.device,
    cls_multilabel: bool,
    cls_threshold: float,
) -> Dict[str, float]:
    model.eval()
    running = {
        "loss_total": 0.0,
        "loss_ce": 0.0,
        "loss_dice": 0.0,
        "loss_focal": 0.0,
        "loss_grad": 0.0,
        "loss_contrastive": 0.0,
        "loss_cls": 0.0,
        "loss_cls_contrastive": 0.0,
    }
    cls_confusion: torch.Tensor | None = None
    tp_per_class: torch.Tensor | None = None
    fp_per_class: torch.Tensor | None = None
    fn_per_class: torch.Tensor | None = None
    exact_match_correct = 0
    exact_match_total = 0
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        class_labels = batch.get("class_label")
        class_labels_multi = batch.get("class_label_multi")
        if class_labels is not None:
            class_labels = class_labels.to(device)
        if class_labels_multi is not None:
            class_labels_multi = class_labels_multi.to(device)
        outputs = model(images)
        losses = model.compute_losses(
            outputs["logits"],
            masks,
            projected_features=outputs.get("projected_features"),
            cls_logits=outputs.get("cls_logits"),
            cls_targets=class_labels,
            cls_targets_multi=class_labels_multi,
            cls_embedding=outputs.get("cls_embedding"),
        )
        for k in running:
            running[k] += losses[k].item()

        cls_logits = outputs.get("cls_logits")
        if cls_logits is not None:
            if cls_multilabel and class_labels_multi is not None:
                cls_probs = torch.sigmoid(cls_logits)
                cls_preds = (cls_probs >= float(cls_threshold)).to(torch.int64).detach().cpu()
                cls_tgts = (class_labels_multi > 0.5).to(torch.int64).detach().cpu()

                num_cls = int(cls_preds.shape[1])
                if tp_per_class is None or int(tp_per_class.shape[0]) != num_cls:
                    tp_per_class = torch.zeros(num_cls, dtype=torch.int64)
                    fp_per_class = torch.zeros(num_cls, dtype=torch.int64)
                    fn_per_class = torch.zeros(num_cls, dtype=torch.int64)

                tp_per_class += ((cls_preds == 1) & (cls_tgts == 1)).sum(dim=0)
                fp_per_class += ((cls_preds == 1) & (cls_tgts == 0)).sum(dim=0)
                fn_per_class += ((cls_preds == 0) & (cls_tgts == 1)).sum(dim=0)
                exact_match_correct += int((cls_preds == cls_tgts).all(dim=1).sum().item())
                exact_match_total += int(cls_preds.shape[0])
            elif class_labels is not None:
                cls_preds = cls_logits.argmax(dim=1)
                num_cls = int(cls_logits.shape[1])
                if cls_confusion is None or int(cls_confusion.shape[0]) != num_cls:
                    cls_confusion = torch.zeros((num_cls, num_cls), dtype=torch.int64)
                for t, p in zip(class_labels.detach().cpu(), cls_preds.detach().cpu()):
                    ti = int(t.item())
                    pi = int(p.item())
                    if 0 <= ti < num_cls and 0 <= pi < num_cls:
                        cls_confusion[ti, pi] += 1
        num_batches += 1

    denom = max(num_batches, 1)
    if cls_multilabel and tp_per_class is not None and fp_per_class is not None and fn_per_class is not None:
        cls_metrics = multilabel_metrics_from_counts(
            tp_per_class=tp_per_class,
            fp_per_class=fp_per_class,
            fn_per_class=fn_per_class,
            exact_match_correct=exact_match_correct,
            num_samples=exact_match_total,
        )
        val_cls_accuracy = cls_metrics["exact_match"]
        val_cls_micro_f1 = cls_metrics["micro_f1"]
        val_cls_macro_f1 = cls_metrics["macro_f1"]
    else:
        cls_metrics = (
            classification_metrics_from_confusion(cls_confusion)
            if cls_confusion is not None
            else {"accuracy": 0.0, "macro_f1": 0.0}
        )
        val_cls_accuracy = cls_metrics["accuracy"]
        val_cls_micro_f1 = cls_metrics["macro_f1"]
        val_cls_macro_f1 = cls_metrics["macro_f1"]

    return {
        "val_loss": running["loss_total"] / denom,
        "val_loss_ce": running["loss_ce"] / denom,
        "val_loss_dice": running["loss_dice"] / denom,
        "val_loss_focal": running["loss_focal"] / denom,
        "val_loss_grad": running["loss_grad"] / denom,
        "val_loss_contrastive": running["loss_contrastive"] / denom,
        "val_loss_cls": running["loss_cls"] / denom,
        "val_loss_cls_contrastive": running["loss_cls_contrastive"] / denom,
        "val_cls_accuracy": val_cls_accuracy,
        "val_cls_micro_f1": val_cls_micro_f1,
        "val_cls_macro_f1": val_cls_macro_f1,
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
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    model_cfg = cfg.get("model", {})
    dent_cfg = model_cfg.get("dent_classification", {})
    cls_cfg = cfg.get("training", {}).get("loss", {}).get("classification", {})
    cls_enabled = bool(cls_cfg.get("enabled", False))
    cls_multilabel = cls_enabled and bool(cls_cfg.get("multilabel", False))
    cls_threshold = float(cls_cfg.get("threshold", 0.5))

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
        class_labels_file=cfg["dataset"].get("class_labels_file"),
        default_class_label=int(cfg["dataset"].get("default_class_label", 0)),
        infer_class_from_mask=bool(cfg["dataset"].get("infer_class_from_mask", False)),
        multilabel_classification=cls_multilabel,
        num_dent_classes=int(dent_cfg.get("num_classes", 0)) if cls_multilabel else None,
    )
    print(f"[Info] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print("[Stage] Model loading...")
    model = DamageSegmentor(
        num_classes=model_cfg["num_classes"],
        pretrained_model_name=model_cfg["pretrained_model_name"],
        loss_config=cfg["training"].get("loss", {}),
    ).to(device)

    optimizer = build_optimizer(model=model, training_cfg=cfg["training"])
    scheduler, scheduler_step_on = build_scheduler(
        optimizer=optimizer,
        training_cfg=cfg["training"],
        epochs=cfg["training"].get("epochs", 25),
    )

    output_dir = Path(cfg["training"].get("output_dir", "outputs/mask2former_tiny"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_enabled = cfg["training"].get("tensorboard", True)
    tb_dir = Path(cfg["training"].get("tensorboard_dir", output_dir / "tensorboard"))
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_enabled else None

    best_val_loss = float("inf")
    history = []
    vis_every = cfg["training"].get("visualize_every", 10)
    vis_samples = cfg["training"].get("visualize_samples", 3)

    early_stopping_cfg = cfg["training"].get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    es_counter = 0
    es_best_score = None

    if early_stopping_enabled:
        es_monitor = early_stopping_cfg.get("monitor", "val_loss")
        es_patience = int(early_stopping_cfg.get("patience", 5))
        es_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
        es_mode = early_stopping_cfg.get("mode", "min")
        es_best_score = float("inf") if es_mode == "min" else float("-inf")
        print(
            f"[Info] Early stopping enabled: monitor='{es_monitor}', "
            f"patience={es_patience}, min_delta={es_min_delta}, mode='{es_mode}'"
        )

    start_epoch = 1
    # Resume from best.pt if it exists
    best_ckpt = output_dir / "best.pt"
    if not args.no_resume and best_ckpt.exists():
        print(f"[Info] Resuming from checkpoint: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and scheduler is not None and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        if "history" in checkpoint:
            history = checkpoint["history"]
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
        print(f"[Info] Resumed at epoch {start_epoch}")

    if writer is not None:
        print(f"[Info] TensorBoard logging enabled at: {tb_dir}")
    if scheduler is not None:
        print(
            f"[Info] Scheduler enabled: {cfg['training']['scheduler']['name']} "
            f"(step on: {scheduler_step_on})"
        )

    epochs = cfg["training"].get("epochs", 25)
    max_train_batches = args.max_train_batches
    if max_train_batches is not None:
        print(f"[Info] max_train_batches={max_train_batches} (smoke test: partial epoch per step)")
    print("[Stage] Training...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = {
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_dice": 0.0,
            "loss_focal": 0.0,
            "loss_grad": 0.0,
            "loss_contrastive": 0.0,
            "loss_cls": 0.0,
            "loss_cls_contrastive": 0.0,
        }
        cls_confusion: torch.Tensor | None = None
        tp_per_class: torch.Tensor | None = None
        fp_per_class: torch.Tensor | None = None
        fn_per_class: torch.Tensor | None = None
        exact_match_correct = 0
        exact_match_total = 0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            class_labels = batch.get("class_label")
            class_labels_multi = batch.get("class_label_multi")
            if class_labels is not None:
                class_labels = class_labels.to(device)
            if class_labels_multi is not None:
                class_labels_multi = class_labels_multi.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = model.compute_losses(
                outputs["logits"],
                masks,
                projected_features=outputs.get("projected_features"),
                cls_logits=outputs.get("cls_logits"),
                cls_targets=class_labels,
                cls_targets_multi=class_labels_multi,
                cls_embedding=outputs.get("cls_embedding"),
            )
            losses["loss_total"].backward()
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()

            cls_logits = outputs.get("cls_logits")
            if cls_logits is not None:
                if cls_multilabel and class_labels_multi is not None:
                    cls_probs = torch.sigmoid(cls_logits)
                    cls_preds = (cls_probs >= float(cls_threshold)).to(torch.int64).detach().cpu()
                    cls_tgts = (class_labels_multi > 0.5).to(torch.int64).detach().cpu()

                    num_cls = int(cls_preds.shape[1])
                    if tp_per_class is None or int(tp_per_class.shape[0]) != num_cls:
                        tp_per_class = torch.zeros(num_cls, dtype=torch.int64)
                        fp_per_class = torch.zeros(num_cls, dtype=torch.int64)
                        fn_per_class = torch.zeros(num_cls, dtype=torch.int64)

                    tp_per_class += ((cls_preds == 1) & (cls_tgts == 1)).sum(dim=0)
                    fp_per_class += ((cls_preds == 1) & (cls_tgts == 0)).sum(dim=0)
                    fn_per_class += ((cls_preds == 0) & (cls_tgts == 1)).sum(dim=0)
                    exact_match_correct += int((cls_preds == cls_tgts).all(dim=1).sum().item())
                    exact_match_total += int(cls_preds.shape[0])
                elif class_labels is not None:
                    cls_preds = cls_logits.argmax(dim=1)
                    num_cls = int(cls_logits.shape[1])
                    if cls_confusion is None or int(cls_confusion.shape[0]) != num_cls:
                        cls_confusion = torch.zeros((num_cls, num_cls), dtype=torch.int64)
                    for t, p in zip(class_labels.detach().cpu(), cls_preds.detach().cpu()):
                        ti = int(t.item())
                        pi = int(p.item())
                        if 0 <= ti < num_cls and 0 <= pi < num_cls:
                            cls_confusion[ti, pi] += 1
            num_batches += 1
            progress.set_postfix(
                loss=f"{losses['loss_total'].item():.4f}",
                ce=f"{losses['loss_ce'].item():.4f}",
                dice=f"{losses['loss_dice'].item():.4f}",
                focal=f"{losses['loss_focal'].item():.4f}",
                grad=f"{losses['loss_grad'].item():.4f}",
                ctr=f"{losses['loss_contrastive'].item():.4f}",
                cls=f"{losses['loss_cls'].item():.4f}",
                cls_ctr=f"{losses['loss_cls_contrastive'].item():.4f}",
            )
            if max_train_batches is not None and num_batches >= max_train_batches:
                break

        train_metrics = {k: v / max(num_batches, 1) for k, v in running.items()}
        if cls_multilabel and tp_per_class is not None and fp_per_class is not None and fn_per_class is not None:
            cls_train_metrics = multilabel_metrics_from_counts(
                tp_per_class=tp_per_class,
                fp_per_class=fp_per_class,
                fn_per_class=fn_per_class,
                exact_match_correct=exact_match_correct,
                num_samples=exact_match_total,
            )
            train_metrics["cls_accuracy"] = cls_train_metrics["exact_match"]
            train_metrics["cls_micro_f1"] = cls_train_metrics["micro_f1"]
            train_metrics["cls_macro_f1"] = cls_train_metrics["macro_f1"]
        else:
            cls_train_metrics = (
                classification_metrics_from_confusion(cls_confusion)
                if cls_confusion is not None
                else {"accuracy": 0.0, "macro_f1": 0.0}
            )
            train_metrics["cls_accuracy"] = cls_train_metrics["accuracy"]
            train_metrics["cls_micro_f1"] = cls_train_metrics["macro_f1"]
            train_metrics["cls_macro_f1"] = cls_train_metrics["macro_f1"]

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            cls_multilabel=cls_multilabel,
            cls_threshold=cls_threshold,
        )

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
            if "loss_contrastive" in train_metrics:
                writer.add_scalar("train/loss_contrastive", train_metrics["loss_contrastive"], epoch)
            if "loss_cls" in train_metrics:
                writer.add_scalar("train/loss_cls", train_metrics["loss_cls"], epoch)
            if "loss_cls_contrastive" in train_metrics:
                writer.add_scalar("train/loss_cls_contrastive", train_metrics["loss_cls_contrastive"], epoch)
            writer.add_scalar("train/cls_accuracy", train_metrics["cls_accuracy"], epoch)
            writer.add_scalar("train/cls_micro_f1", train_metrics["cls_micro_f1"], epoch)
            writer.add_scalar("train/cls_macro_f1", train_metrics["cls_macro_f1"], epoch)
            writer.add_scalar("val/loss_total", val_metrics["val_loss"], epoch)
            writer.add_scalar("val/loss_ce", val_metrics["val_loss_ce"], epoch)
            writer.add_scalar("val/loss_dice", val_metrics["val_loss_dice"], epoch)
            writer.add_scalar("val/loss_focal", val_metrics["val_loss_focal"], epoch)
            writer.add_scalar("val/loss_grad", val_metrics["val_loss_grad"], epoch)
            writer.add_scalar("val/loss_contrastive", val_metrics["val_loss_contrastive"], epoch)
            writer.add_scalar("val/loss_cls", val_metrics["val_loss_cls"], epoch)
            writer.add_scalar("val/loss_cls_contrastive", val_metrics["val_loss_cls_contrastive"], epoch)
            writer.add_scalar("val/cls_accuracy", val_metrics["val_cls_accuracy"], epoch)
            writer.add_scalar("val/cls_micro_f1", val_metrics["val_cls_micro_f1"], epoch)
            writer.add_scalar("val/cls_macro_f1", val_metrics["val_cls_macro_f1"], epoch)
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

        if early_stopping_enabled:
            score = val_metrics.get(es_monitor)
            if score is None:
                print(f"[Warning] Early stopping monitor '{es_monitor}' not found in validation metrics. Skipping.")
            else:
                is_better = False
                if es_mode == "min":
                    if score < es_best_score - es_min_delta:
                        is_better = True
                else:  # max
                    if score > es_best_score + es_min_delta:
                        is_better = True

                if is_better:
                    es_best_score = score
                    es_counter = 0
                else:
                    es_counter += 1

                if es_counter >= es_patience:
                    print(f"[Info] Early stopping triggered at epoch {epoch}")
                    break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    train()
