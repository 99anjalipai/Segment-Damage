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


def average_precision_from_scores(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().cpu().float()
    targets = targets.detach().cpu().to(torch.int64)
    positives = int((targets == 1).sum().item())
    if positives <= 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    y_sorted = targets[order]
    tp_cum = torch.cumsum((y_sorted == 1).to(torch.float32), dim=0)
    ranks = torch.arange(1, y_sorted.numel() + 1, dtype=torch.float32)
    precision_at_k = tp_cum / ranks
    pos_mask = y_sorted == 1
    ap = precision_at_k[pos_mask].mean()
    return float(ap.item())


def multilabel_per_class_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Dict[str, list[float] | float]:
    if probs.numel() == 0 or targets.numel() == 0:
        return {
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
            "per_class_ap": [],
            "mAP": 0.0,
        }

    probs = probs.detach().cpu().float()
    targets = targets.detach().cpu().to(torch.int64)
    preds = (probs >= float(threshold)).to(torch.int64)
    num_classes = int(probs.shape[1])

    per_precision: list[float] = []
    per_recall: list[float] = []
    per_f1: list[float] = []
    per_ap: list[float] = []

    for cls_idx in range(num_classes):
        p = preds[:, cls_idx]
        t = targets[:, cls_idx]

        tp = int(((p == 1) & (t == 1)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

        ap = average_precision_from_scores(probs[:, cls_idx], t)
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)
        per_ap.append(ap)

    m_ap = float(sum(per_ap) / max(len(per_ap), 1))
    return {
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "per_class_ap": per_ap,
        "mAP": m_ap,
    }


def embedding_separation_metrics(
    embeddings: torch.Tensor,
    num_classes: int,
    cls_targets: torch.Tensor | None = None,
    cls_targets_multi: torch.Tensor | None = None,
) -> Dict[str, object]:
    if embeddings.numel() == 0 or embeddings.ndim != 2:
        return {
            "cls_embedding_dim": 0,
            "cls_centroid_counts": [],
            "cls_centroid_cosine_similarity": [],
            "cls_centroid_cosine_distance": [],
            "cls_inter_class_centroid_distance_mean": 0.0,
            "cls_per_class_intra_distance": [],
        }

    emb = embeddings.detach().cpu().float()
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    class_masks: list[torch.Tensor] = []
    if cls_targets_multi is not None:
        tgt_m = (cls_targets_multi.detach().cpu() > 0.5)
        for cls_idx in range(num_classes):
            class_masks.append(tgt_m[:, cls_idx])
    elif cls_targets is not None:
        tgt = cls_targets.detach().cpu().view(-1)
        for cls_idx in range(num_classes):
            class_masks.append(tgt == cls_idx)
    else:
        class_masks = [torch.zeros(emb.shape[0], dtype=torch.bool) for _ in range(num_classes)]

    centroids: list[torch.Tensor | None] = []
    counts: list[int] = []
    intra_dist: list[float] = []
    for cls_idx in range(num_classes):
        mask = class_masks[cls_idx]
        count = int(mask.sum().item())
        counts.append(count)
        if count <= 0:
            centroids.append(None)
            intra_dist.append(0.0)
            continue

        cls_emb = emb[mask]
        centroid = cls_emb.mean(dim=0)
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
        centroids.append(centroid)

        mean_cos = torch.clamp(cls_emb @ centroid, min=-1.0, max=1.0).mean().item()
        intra_dist.append(float(1.0 - mean_cos))

    sim_matrix: list[list[float | None]] = []
    dist_matrix: list[list[float | None]] = []
    inter_dists: list[float] = []
    for i in range(num_classes):
        sim_row: list[float | None] = []
        dist_row: list[float | None] = []
        for j in range(num_classes):
            ci = centroids[i]
            cj = centroids[j]
            if ci is None or cj is None:
                sim_row.append(None)
                dist_row.append(None)
                continue
            sim = float(torch.clamp(torch.dot(ci, cj), min=-1.0, max=1.0).item())
            dist = float(1.0 - sim)
            sim_row.append(sim)
            dist_row.append(dist)
            if i < j:
                inter_dists.append(dist)
        sim_matrix.append(sim_row)
        dist_matrix.append(dist_row)

    inter_mean = float(sum(inter_dists) / max(len(inter_dists), 1))
    return {
        "cls_embedding_dim": int(emb.shape[1]),
        "cls_centroid_counts": counts,
        "cls_centroid_cosine_similarity": sim_matrix,
        "cls_centroid_cosine_distance": dist_matrix,
        "cls_inter_class_centroid_distance_mean": inter_mean,
        "cls_per_class_intra_distance": intra_dist,
    }


def load_class_label_index_to_name(class_labels_file: str | Path | None) -> Dict[str, str]:
    if not class_labels_file:
        return {}

    path = Path(class_labels_file)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    index_to_name = metadata.get("index_to_name", {})
    if not isinstance(index_to_name, dict):
        return {}
    return {str(k): str(v) for k, v in index_to_name.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 baseline evaluation.")
    parser.add_argument("--config", type=str, default="configs/mask2former_tiny.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/mask2former_tiny/best.pt")
    parser.add_argument("--tiny-area-threshold", type=int, default=1500)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/mask2former_tiny/eval",
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
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        dest="max_batches",
        help="Evaluate only this many batches (smoke tests; omit for full split).",
    )
    return parser.parse_args()


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    img = image * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).detach().cpu().numpy()

def save_tsne_visualization(
    embeddings: torch.Tensor,
    targets: torch.Tensor | None,
    targets_multi: torch.Tensor | None,
    num_classes: int,
    output_dir: Path,
    cls_index_to_name: Dict[str, str] | None = None
) -> None:
    if embeddings is None or embeddings.numel() == 0 or embeddings.shape[0] < 2:
        return

    emb_np = embeddings.detach().cpu().numpy()
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    if targets_multi is not None:
        tgt_np = targets_multi.detach().cpu().numpy()
        for cls_idx in range(num_classes):
            mask = tgt_np[:, cls_idx] > 0.5
            if not mask.any():
                continue
            name = cls_index_to_name.get(str(cls_idx), f"Class {cls_idx}") if cls_index_to_name else f"Class {cls_idx}"
            ax.scatter(reduced[mask, 0], reduced[mask, 1], label=name, alpha=0.6)
    elif targets is not None:
        tgt_np = targets.detach().cpu().numpy()
        for cls_idx in range(num_classes):
            mask = tgt_np == cls_idx
            if not mask.any():
                continue
            name = cls_index_to_name.get(str(cls_idx), f"Class {cls_idx}") if cls_index_to_name else f"Class {cls_idx}"
            ax.scatter(reduced[mask, 0], reduced[mask, 1], label=name, alpha=0.6)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)

    ax.set_title("t-SNE of Class Embeddings")
    if targets_multi is not None or targets is not None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "embeddings_tsne.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

@torch.no_grad()
def save_eval_visualizations(
    model: DamageSegmentor,
    dataset,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 5,
) -> None:
    if max_samples <= 0:
        return

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
    cls_multilabel: bool,
    cls_threshold: float,
    cls_num_classes: int,
    max_batches: int | None = None,
) -> tuple[Dict[str, float], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    model.eval()
    iou_numer = torch.zeros(num_classes, dtype=torch.float64)
    iou_denom = torch.zeros(num_classes, dtype=torch.float64)

    tiny_tp = 0
    tiny_fn = 0
    cls_confusion: torch.Tensor | None = None
    tp_per_class: torch.Tensor | None = None
    fp_per_class: torch.Tensor | None = None
    fn_per_class: torch.Tensor | None = None
    exact_match_correct = 0
    exact_match_total = 0
    all_cls_probs: list[torch.Tensor] = []
    all_cls_targets: list[torch.Tensor] = []
    all_cls_embeddings: list[torch.Tensor] = []
    all_cls_targets_single: list[torch.Tensor] = []
    all_cls_targets_multi: list[torch.Tensor] = []

    n_batch = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        logits = outputs["logits"]
        preds = logits.argmax(dim=1)

        class_labels = batch.get("class_label")
        class_labels_multi = batch.get("class_label_multi")
        cls_logits = outputs.get("cls_logits")
        cls_embedding = outputs.get("cls_embedding")
        if cls_logits is not None:
            if cls_multilabel and class_labels_multi is not None:
                cls_probs = torch.sigmoid(cls_logits)
                cls_preds = (cls_probs >= float(cls_threshold)).to(torch.int64).detach().cpu()
                cls_tgts = (class_labels_multi > 0.5).to(torch.int64).detach().cpu()
                all_cls_probs.append(cls_probs.detach().cpu())
                all_cls_targets.append(cls_tgts)
                if cls_embedding is not None:
                    all_cls_embeddings.append(cls_embedding.detach().cpu())
                    all_cls_targets_multi.append(cls_tgts)

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
                cls_preds = cls_logits.argmax(dim=1).detach().cpu()
                cls_targets = class_labels.detach().cpu()
                if cls_embedding is not None:
                    all_cls_embeddings.append(cls_embedding.detach().cpu())
                    all_cls_targets_single.append(cls_targets)
                num_cls = int(cls_logits.shape[1])
                if cls_confusion is None or int(cls_confusion.shape[0]) != num_cls:
                    cls_confusion = torch.zeros((num_cls, num_cls), dtype=torch.int64)
                for t, p in zip(cls_targets, cls_preds):
                    ti = int(t.item())
                    pi = int(p.item())
                    if 0 <= ti < num_cls and 0 <= pi < num_cls:
                        cls_confusion[ti, pi] += 1

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

        n_batch += 1
        if max_batches is not None and n_batch >= max_batches:
            break

    iou_per_class = (iou_numer / torch.clamp(iou_denom, min=1.0)).tolist()
    miou = float(sum(iou_per_class) / len(iou_per_class))
    det_l = float(tiny_tp / max(tiny_tp + tiny_fn, 1))
    f1_from_miou = float((2 * miou) / max(miou + 1.0, 1e-8))
    if cls_multilabel and tp_per_class is not None and fp_per_class is not None and fn_per_class is not None:
        cls_metrics = multilabel_metrics_from_counts(
            tp_per_class=tp_per_class,
            fp_per_class=fp_per_class,
            fn_per_class=fn_per_class,
            exact_match_correct=exact_match_correct,
            num_samples=exact_match_total,
        )
        cls_accuracy = cls_metrics["exact_match"]
        cls_micro_f1 = cls_metrics["micro_f1"]
        cls_macro_f1 = cls_metrics["macro_f1"]
        if all_cls_probs and all_cls_targets:
            concat_probs = torch.cat(all_cls_probs, dim=0)
            concat_targets = torch.cat(all_cls_targets, dim=0)
            cls_per_class = multilabel_per_class_metrics(
                probs=concat_probs,
                targets=concat_targets,
                threshold=cls_threshold,
            )
        else:
            cls_per_class = {
                "per_class_precision": [],
                "per_class_recall": [],
                "per_class_f1": [],
                "per_class_ap": [],
                "mAP": 0.0,
            }
    else:
        cls_metrics = (
            classification_metrics_from_confusion(cls_confusion)
            if cls_confusion is not None
            else {"accuracy": 0.0, "macro_f1": 0.0}
        )
        cls_accuracy = cls_metrics["accuracy"]
        cls_micro_f1 = cls_metrics["macro_f1"]
        cls_macro_f1 = cls_metrics["macro_f1"]
        cls_per_class = {
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
            "per_class_ap": [],
            "mAP": 0.0,
        }

    if all_cls_embeddings:
        concat_emb = torch.cat(all_cls_embeddings, dim=0)
        concat_targets_single = (
            torch.cat(all_cls_targets_single, dim=0)
            if all_cls_targets_single
            else None
        )
        concat_targets_multi = (
            torch.cat(all_cls_targets_multi, dim=0)
            if all_cls_targets_multi
            else None
        )
        cls_embedding_stats = embedding_separation_metrics(
            embeddings=concat_emb,
            num_classes=int(cls_num_classes),
            cls_targets=concat_targets_single,
            cls_targets_multi=concat_targets_multi,
        )
        cls_centroid_distance_mean = cls_embedding_stats["cls_inter_class_centroid_distance_mean"]
        cls_intra_distance = cls_embedding_stats["cls_per_class_intra_distance"]
        cls_embedding_dim = cls_embedding_stats["cls_embedding_dim"]
        cls_centroid_counts = cls_embedding_stats["cls_centroid_counts"]
        cls_centroid_similarity = cls_embedding_stats["cls_centroid_cosine_similarity"]
        cls_centroid_distance = cls_embedding_stats["cls_centroid_cosine_distance"]
    else:
        concat_emb = None
        concat_targets_single = None
        concat_targets_multi = None
        cls_embedding_dim = 0
        cls_centroid_counts = []
        cls_centroid_similarity = []
        cls_centroid_distance = []
        cls_centroid_distance_mean = 0.0
        cls_intra_distance = []

    metrics_dict = {
        "mIoU": miou,
        "IoU_per_class": iou_per_class,
        "F1_proxy": f1_from_miou,
        "DET_l": det_l,
        "tiny_true_positive": tiny_tp,
        "tiny_false_negative": tiny_fn,
        "cls_accuracy": cls_accuracy,
        "cls_micro_f1": cls_micro_f1,
        "cls_macro_f1": cls_macro_f1,
        "cls_per_class_precision": cls_per_class["per_class_precision"],
        "cls_per_class_recall": cls_per_class["per_class_recall"],
        "cls_per_class_f1": cls_per_class["per_class_f1"],
        "cls_per_class_ap": cls_per_class["per_class_ap"],
        "cls_mAP": cls_per_class["mAP"],
        "cls_embedding_dim": cls_embedding_dim,
        "cls_centroid_counts": cls_centroid_counts,
        "cls_centroid_cosine_similarity": cls_centroid_similarity,
        "cls_centroid_cosine_distance": cls_centroid_distance,
        "cls_inter_class_centroid_distance_mean": cls_centroid_distance_mean,
        "cls_per_class_intra_distance": cls_intra_distance,
    }
    
    return metrics_dict, concat_emb, concat_targets_single, concat_targets_multi


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})
    output_dir = Path(training_cfg.get("output_dir", "outputs/mask2former_tiny"))

    # If user didn't override defaults, follow the configured training output directory.
    checkpoint_path = Path(args.checkpoint)
    if args.checkpoint == "outputs/mask2former_tiny/best.pt":
        candidate_best = output_dir / "best.pt"
        candidate_last = output_dir / "last.pt"
        if candidate_best.exists():
            checkpoint_path = candidate_best
        elif candidate_last.exists():
            checkpoint_path = candidate_last

    results_dir_arg = Path(args.results_dir)
    if args.results_dir == "outputs/mask2former_tiny/eval":
        results_dir_arg = output_dir / "eval"

    model_cfg = cfg.get("model", {})
    dent_cfg = model_cfg.get("dent_classification", {})
    cls_cfg = cfg.get("training", {}).get("loss", {}).get("classification", {})
    cls_enabled = bool(cls_cfg.get("enabled", False))
    cls_multilabel = cls_enabled and bool(cls_cfg.get("multilabel", False))
    cls_threshold = float(cls_cfg.get("threshold", 0.5))
    cls_index_to_name = load_class_label_index_to_name(cfg.get("dataset", {}).get("class_labels_file"))

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
        class_labels_file=cfg["dataset"].get("class_labels_file"),
        default_class_label=int(cfg["dataset"].get("default_class_label", 0)),
        infer_class_from_mask=bool(cfg["dataset"].get("infer_class_from_mask", False)),
        multilabel_classification=cls_multilabel,
        num_dent_classes=int(dent_cfg.get("num_classes", 0)) if cls_multilabel else None,
    )
    from torch.utils.data import DataLoader
    print("Loading dataset and creating dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )
    print(f"[Info] Dataset loaded with {len(dataset)} samples. Evaluating on split: {args.split}")

    model = DamageSegmentor(
        num_classes=model_cfg["num_classes"],
        pretrained_model_name=model_cfg["pretrained_model_name"],
        loss_config=cfg["training"].get("loss", {}),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] Model total parameters: {total_params}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Pass --checkpoint explicitly or verify training.output_dir in your config."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"[Stage] Running evaluation on {args.split} split with {len(loader)} batches...")
    metrics, concat_emb, concat_targets_single, concat_targets_multi = evaluate(
        model=model,
        loader=loader,
        device=device,
        tiny_area_threshold=args.tiny_area_threshold,
        num_classes=cfg["model"]["num_classes"],
        cls_multilabel=cls_multilabel,
        cls_threshold=cls_threshold,
        cls_num_classes=int(dent_cfg.get("num_classes", 0)),
        max_batches=args.max_batches,
    )
    if cls_index_to_name:
        metrics["cls_index_to_name"] = cls_index_to_name
    print(json.dumps(metrics, indent=2))

    # Save metrics to results dir
    results_dir = results_dir_arg
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Info] Metrics saved to {metrics_path}")

    if concat_emb is not None:
        print(f"[Stage] Saving t-SNE embedding visualization...")
        save_tsne_visualization(
            embeddings=concat_emb,
            targets=concat_targets_single,
            targets_multi=concat_targets_multi,
            num_classes=int(dent_cfg.get("num_classes", 0)),
            output_dir=results_dir,
            cls_index_to_name=cls_index_to_name
        )
        print(f"[Info] t-SNE plot saved as {results_dir / 'visualizations' / 'embeddings_tsne.png'}")

    # Save visualizations
    print(f"[Stage] Saving evaluation visualizations to {results_dir / 'visualizations'} ...")
    save_eval_visualizations(model, dataset, device, results_dir, max_samples=args.visualize_samples)
    print(f"[Info] Visualization saved as {results_dir / 'visualizations' / 'eval_samples.png'}")


if __name__ == "__main__":
    main()
