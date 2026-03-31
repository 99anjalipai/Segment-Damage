import os
import json
import argparse
import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_split_ids(split_json_path):
    with open(split_json_path, "r") as f:
        data = json.load(f)
    return data["items"]


def find_file(search_dirs, sample_id, exts=(".png", ".jpg", ".jpeg")):
    for folder in search_dirs:
        for ext in exts:
            path = os.path.join(folder, f"{sample_id}{ext}")
            if os.path.exists(path):
                return path
    return None


def compute_iou(pred_mask, true_mask, cls):
    pred = (pred_mask == cls)
    true = (true_mask == cls)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return inter / union if union > 0 else 1.0


def f1_from_iou(iou):
    return (2 * iou) / (1 + iou) if iou > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sod-root", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tiny-thresh", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    image_search_dirs = [
        os.path.join(args.sod_root, "CarDD-TR", "CarDD-TR-Image"),
        os.path.join(args.sod_root, "CarDD-TE", "CarDD-TE-Image"),
    ]
    mask_search_dirs = [
        os.path.join(args.sod_root, "CarDD-TR", "CarDD-TR-Mask"),
        os.path.join(args.sod_root, "CarDD-TE", "CarDD-TE-Mask"),
    ]

    split_ids = load_split_ids(os.path.join(args.splits_dir, f"{args.split}.json"))
    model = YOLO(args.checkpoint)

    ious_bg = []
    ious_fg = []
    tiny_tp = 0
    tiny_fn = 0
    tiny_samples = 0

    for sample_id in split_ids:
        img_path = find_file(image_search_dirs, sample_id)
        mask_path = find_file(mask_search_dirs, sample_id, exts=(".png",))

        if img_path is None or mask_path is None:
            print(f"Skipping {sample_id} (missing file)")
            continue

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"Skipping {sample_id} (mask unreadable)")
            continue

        gt = (gt > 0).astype(np.uint8)

        results = model.predict(
            source=img_path,
            imgsz=cfg["model"]["imgsz"],
            conf=cfg["evaluation"]["conf"],
            iou=cfg["evaluation"]["iou"],
            verbose=False
        )

        pred = np.zeros_like(gt, dtype=np.uint8)

        r = results[0]
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for m in masks:
                m_resized = cv2.resize(
                    m,
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                pred[m_resized > 0.5] = 1

        iou_bg = compute_iou(pred, gt, 0)
        iou_fg = compute_iou(pred, gt, 1)
        ious_bg.append(iou_bg)
        ious_fg.append(iou_fg)

        gt_fg = int((gt > 0).sum())

        if gt_fg < args.tiny_thresh:
            tiny_samples += 1
            print(f"TINY sample: {sample_id}, area={gt_fg}")

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

    metrics = {
        "split": args.split,
        "mIoU": miou,
        "IoU_per_class": [mean_bg, mean_fg],
        "F1_proxy": f1_proxy,
        "DET_l": det_l,
        "tiny_area_threshold": args.tiny_thresh,
        "tiny_samples": tiny_samples,
        "tiny_true_positive": tiny_tp,
        "tiny_false_negative": tiny_fn,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()