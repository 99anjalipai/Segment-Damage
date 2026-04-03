import os
import yaml
import subprocess
from pathlib import Path
import pandas as pd
import json

BASE_CONFIG = "configs/yolov8seg.yaml"
SWEEP_CONFIG = "configs/yolov8seg_optimizations.yaml"

SOD_ROOT = "/content/drive/MyDrive/SegmentDamage/CarDD_dataset/CarDD_SOD"
SPLITS_DIR = "data/splits"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def run_command(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def find_latest_run_dir(output_root: Path, exp_name: str) -> Path:
    runs_base = Path("runs/segment/outputs") / output_root.name
    all_runs = sorted(runs_base.glob(f"{exp_name}*"), key=os.path.getmtime)

    if not all_runs:
        raise RuntimeError(f"No YOLO run folder found for experiment: {exp_name}")

    latest = all_runs[-1]
    print(f"[INFO] Using run dir: {latest}")
    return latest


def load_metrics(metrics_path: Path) -> dict:
    with open(metrics_path, "r") as f:
        return json.load(f)


def write_leaderboards(rows, output_root: Path):
    if not rows:
        print("[WARN] No rows to summarize.")
        return

    df = pd.DataFrame(rows)

    summary_json = output_root / "comparison_summary.json"
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split].copy()

        global_df = split_df.sort_values(
            by=["mIoU", "F1_proxy", "DET_l"],
            ascending=[False, False, False]
        )
        global_df.to_csv(output_root / f"leaderboard_{split}.csv", index=False)

        tiny_df = split_df.sort_values(
            by=["DET_l", "mIoU", "F1_proxy"],
            ascending=[False, False, False]
        )
        tiny_df.to_csv(output_root / f"leaderboard_tiny_{split}.csv", index=False)

    print(f"[INFO] Wrote summary files to {output_root}")

def main():
    base_cfg = load_yaml(BASE_CONFIG)
    sweep_cfg = load_yaml(SWEEP_CONFIG)

    output_root = Path(sweep_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    generated_cfg_dir = output_root / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    experiments = sweep_cfg["experiments"]

    for exp in experiments:
        name = exp["name"]
        overrides = exp["overrides"]

        print(f"\n===== Running {name} =====")

        # clone base config
        cfg = base_cfg.copy()

        # apply overrides
        cfg["model"]["pretrained_weights"] = overrides["pretrained_weights"]
        cfg["model"]["imgsz"] = overrides["imgsz"]
        cfg["training"]["epochs"] = overrides["epochs"]
        cfg["training"]["batch"] = overrides["batch"]

        # update output name
        cfg["training"]["name"] = name
        cfg["training"]["project"] = str(output_root)

        # save generated config
        gen_cfg_path = output_root / f"{name}.yaml"
        save_yaml(cfg, gen_cfg_path)

        # ===== TRAIN =====
        run_command(
            f"python tools/train_yolov8seg.py --config {gen_cfg_path}"
        )

        # find actual YOLO run folder
        runs_base = Path("runs/segment/outputs") / output_root.name

        all_runs = sorted(runs_base.glob(f"{name}*"), key=os.path.getmtime)
        
        if not all_runs:
            raise RuntimeError(f"No run folder found for {name}")
        
        run_dir = all_runs[-1]  # latest
        
        weights_path = run_dir / "weights" / "best.pt"
        print(f"[INFO] Using weights from: {weights_path}")

        # ===== EVAL (VAL) =====
        for split in sweep_cfg["evaluate_splits"]:
            results_dir = output_root / name / "eval" / split

            run_command(
                f"python tools/evaluate_yolov8seg.py "
                f"--config {gen_cfg_path} "
                f"--checkpoint {weights_path} "
                f"--sod-root {SOD_ROOT} "
                f"--splits-dir {SPLITS_DIR} "
                f"--split {split} "
                f"--results-dir {results_dir} "
                f"--tiny-thresh {sweep_cfg['tiny_area_threshold']} "
                f"--visualize-samples {sweep_cfg['visualize_samples'] if split == sweep_cfg['comparison_split'] else 0}"
            )

            metrics_path = results_dir / "metrics.json"
            metrics = load_metrics(metrics_path)

            summary_rows.append({
                "experiment": name,
                "split": split,
                "weights": str(weights_path),
                "pretrained_weights": overrides["pretrained_weights"],
                "imgsz": overrides["imgsz"],
                "batch": overrides["batch"],
                "epochs": overrides["epochs"],
                "mIoU": metrics["mIoU"],
                "F1_proxy": metrics["F1_proxy"],
                "DET_l": metrics["DET_l"],
                "tiny_samples": metrics["tiny_samples"],
                "tiny_true_positive": metrics["tiny_true_positive"],
                "tiny_false_negative": metrics["tiny_false_negative"],
            })

        write_leaderboards(summary_rows, output_root)


if __name__ == "__main__":
    main()