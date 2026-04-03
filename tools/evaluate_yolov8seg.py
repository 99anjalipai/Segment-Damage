from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-seg on CarDD_SOD.")
    parser.add_argument("--config", required=True, help="Path to YOLO config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained YOLO checkpoint.")
    parser.add_argument("--sod-root", required=True, help="Path to CarDD_SOD root.")
    parser.add_argument("--splits-dir", required=True, help="Directory containing split JSON files.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory to save metrics and visualizations.",
    )
    parser.add_argument(
        "--tiny-thresh",
        type=int,
        default=5000,
        help="Foreground area threshold for tiny-damage DET_l.",
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=5,
        help="Number of qualitative samples to save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for visualization sampling.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_ids(split_json_path: str) -> list[str]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def find_file(search_dirs: list[str], sample_id: str, exts=(".png", ".jpg", ".jpeg")) -> str | None:
    for folder in search_dirs:
        for ext in exts:
            path = os.path.join(folder, f"{sample_id}{ext}")
            if os.path.exists(path):
                return path
    return None


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray, cls: int) -> float:
    pred = pred_mask == cls
    true = true_mask == cls
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return inter / union if union > 0 else 1.0


def f1_from_iou(iou: float) -> float:
    return (2 * iou) / (1 + iou) if iou > 0 else 0.0


def build_prediction_mask(
    model: YOLO,
    img_path: str,
    gt_shape: tuple[int, int],
    imgsz: int,
    conf: float,
    iou: float,
) -> np.ndarray:
    gt_h, gt_w = gt_shape
    pred = np.zeros((gt_h, gt_w), dtype=np.uint8)

    results = model.predict(
        source=img_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    result = results[0]
    if result.masks is None:
        return pred

    masks = result.masks.data.cpu().numpy()
    for mask in masks:
        resized = cv2.resize(mask, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
        pred[resized > 0.5] = 1

    return pred


def load_sample(
    sample_id: str,
    image_search_dirs: list[str],
    mask_search_dirs: list[str],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    img_path = find_file(image_search_dirs, sample_id, exts=(".jpg", ".jpeg", ".png"))
    mask_path = find_file(mask_search_dirs, sample_id, exts=(".png",))

    if img_path is None or mask_path is None:
        print(f"Skipping {sample_id} (missing image or mask)")
        return None

    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or gt is None:
        print(f"Skipping {sample_id} (unreadable image or mask)")
        return None

    gt = (gt > 0).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, gt, img_path


def save_eval_visualizations(
    model: YOLO,
    split_ids: list[str],
    image_search_dirs: list[str],
    mask_search_dirs: list[str],
    cfg: dict,
    output_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    if max_samples <= 0 or len(split_ids) == 0:
        return

    random.seed(seed)
    num_samples = min(max_samples, len(split_ids))
    selected_ids = random.sample(split_ids, num_samples)

    rows = []

    for sample_id in selected_ids:
        loaded = load_sample(sample_id, image_search_dirs, mask_search_dirs)
        if loaded is None:
            continue

        image_rgb, gt, img_path = loaded
        pred = build_prediction_mask(
            model=model,
            img_path=img_path,
            gt_shape=gt.shape,
            imgsz=cfg["model"]["imgsz"],
            conf=cfg["evaluation"]["conf"],
            iou=cfg["evaluation"]["iou"],
        )
        rows.append((sample_id, image_rgb, gt, pred))

    if not rows:
        print("[Warning] No visualization samples could be generated.")
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (sample_id, image_rgb, gt, pred) in enumerate(rows):
        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(f"Input\n{sample_id}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(vis_dir / "eval_samples.png", dpi=150)
    plt.close(fig)


def evaluate(
    model: YOLO,
    split_ids: list[str],
    image_search_dirs: list[str],
    mask_search_dirs: list[str],
    cfg: dict,
    tiny_thresh: int,
) -> dict:
    ious_bg = []
    ious_fg = []
    tiny_tp = 0
    tiny_fn = 0
    tiny_samples = 0

    for sample_id in split_ids:
        loaded = load_sample(sample_id, image_search_dirs, mask_search_dirs)
        if loaded is None:
            continue

        _, gt, img_path = loaded

        pred = build_prediction_mask(
            model=model,
            img_path=img_path,
            gt_shape=gt.shape,
            imgsz=cfg["model"]["imgsz"],
            conf=cfg["evaluation"]["conf"],
            iou=cfg["evaluation"]["iou"],
        )

        iou_bg = compute_iou(pred, gt, 0)
        iou_fg = compute_iou(pred, gt, 1)
        ious_bg.append(iou_bg)
        ious_fg.append(iou_fg)

        gt_fg = int((gt > 0).sum())
        if gt_fg < tiny_thresh:
            tiny_samples += 1
            overlap = np.logical_and(pred > 0, gt > 0).sum()
            if overlap > 0:
                tiny_tp += 1
            else:
                tiny_fn += 1

    mean_bg = float(np.mean(ious_bg)) if ious_bg else 0.0
    mean_fg = float(np.mean(ious_fg)) if ious_fg else 0.0
    miou = (mean_bg + mean_fg) / 2.0
    f1_proxy = f1_from_iou(miou)
    det_l = tiny_tp / (tiny_tp + tiny_fn) if (tiny_tp + tiny_fn) > 0 else 0.0

    return {
        "mIoU": miou,
        "IoU_per_class": [mean_bg, mean_fg],
        "F1_proxy": f1_proxy,
        "DET_l": det_l,
        "tiny_area_threshold": tiny_thresh,
        "tiny_samples": tiny_samples,
        "tiny_true_positive": tiny_tp,
        "tiny_false_negative": tiny_fn,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    image_search_dirs = [
        os.path.join(args.sod_root, "CarDD-TR", "CarDD-TR-Image"),
        os.path.join(args.sod_root, "CarDD-TE", "CarDD-TE-Image"),
    ]
    mask_search_dirs = [
        os.path.join(args.sod_root, "CarDD-TR", "CarDD-TR-Mask"),
        os.path.join(args.sod_root, "CarDD-TE", "CarDD-TE-Mask"),
    ]

    split_json_path = os.path.join(args.splits_dir, f"{args.split}.json")
    split_ids = load_split_ids(split_json_path)

    model = YOLO(args.checkpoint)

    metrics = evaluate(
        model=model,
        split_ids=split_ids,
        image_search_dirs=image_search_dirs,
        mask_search_dirs=mask_search_dirs,
        cfg=cfg,
        tiny_thresh=args.tiny_thresh,
    )
    metrics["split"] = args.split

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[Info] Metrics saved to {metrics_path}")

    print(f"[Stage] Saving evaluation visualizations to {results_dir / 'visualizations'} ...")
    save_eval_visualizations(
        model=model,
        split_ids=split_ids,
        image_search_dirs=image_search_dirs,
        mask_search_dirs=mask_search_dirs,
        cfg=cfg,
        output_dir=results_dir,
        max_samples=args.visualize_samples,
        seed=args.seed,
    )
    print(f"[Info] Visualization saved to {results_dir / 'visualizations' / 'eval_samples.png'}")


if __name__ == "__main__":
    main()