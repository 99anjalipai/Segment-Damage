from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline optimization experiments by training and evaluating each setup "
            "on train/val/test splits."
        )
    )
    parser.add_argument("--base-config", type=str, default="configs/week1_unet.yaml")
    parser.add_argument(
        "--experiments-config",
        type=str,
        default="configs/baseline_optimizations.yaml",
        help="YAML file describing optimization experiments.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/baseline_optimizations",
        help="Root directory where per-experiment outputs will be stored.",
    )
    parser.add_argument(
        "--evaluate-splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining experiments if one fails.",
    )
    parser.add_argument(
        "--comparison-split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Only this split stores qualitative visualizations per experiment.",
    )
    parser.add_argument(
        "--comparison-visualize-samples",
        type=int,
        default=3,
        help="Number of qualitative samples to save for comparison split (capped to 3).",
    )
    parser.add_argument(
        "--tiny-area-threshold",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Override training epochs for all experiments (useful for smoke tests).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_metrics(metrics_file: Path) -> Dict[str, Any]:
    with metrics_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_config(
    base_cfg: Dict[str, Any],
    exp_name: str,
    exp_overrides: Dict[str, Any],
    output_root: Path,
    epochs_override: int | None = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = deep_update(cfg, exp_overrides)

    exp_out = output_root / exp_name
    cfg.setdefault("training", {})
    cfg["training"]["output_dir"] = str(exp_out)
    cfg["training"]["tensorboard_dir"] = str(exp_out / "tensorboard")
    if epochs_override is not None:
        cfg["training"]["epochs"] = int(epochs_override)
    return cfg


def build_leaderboard(all_results: Dict[str, Any], primary_split: str = "val") -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    experiments = all_results.get("experiments", {})
    for exp_name, exp_record in experiments.items():
        if exp_record.get("status") != "completed":
            continue

        split_metrics = exp_record.get("splits", {}).get(primary_split, {})
        det_l = float(split_metrics.get("DET_l", -1.0))
        miou = float(split_metrics.get("mIoU", -1.0))
        f1_proxy = float(split_metrics.get("F1_proxy", -1.0))
        rows.append(
            {
                "experiment": exp_name,
                "split": primary_split,
                "DET_l": det_l,
                "mIoU": miou,
                "F1_proxy": f1_proxy,
            }
        )

    # Priority: DET_l first, then mIoU, then F1 proxy.
    rows.sort(key=lambda r: (r["DET_l"], r["mIoU"], r["F1_proxy"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def update_summary_fields(all_results: Dict[str, Any], primary_split: str) -> None:
    leaderboard = build_leaderboard(all_results=all_results, primary_split=primary_split)
    all_results["leaderboard"] = leaderboard
    if leaderboard:
        all_results["best_experiment"] = leaderboard[0]["experiment"]
    else:
        all_results["best_experiment"] = None


def main() -> None:
    args = parse_args()

    base_config_path = ROOT / args.base_config
    experiments_config_path = ROOT / args.experiments_config
    output_root = ROOT / args.output_root

    base_cfg = load_yaml(base_config_path)
    exp_cfg = load_yaml(experiments_config_path)

    experiments = exp_cfg.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in experiments config.")

    output_root.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir = output_root / "generated_configs"

    all_results: Dict[str, Any] = {
        "base_config": str(base_config_path),
        "experiments_config": str(experiments_config_path),
        "evaluate_splits": args.evaluate_splits,
        "comparison_split": args.comparison_split,
        "epochs_override": args.epochs_override,
        "experiments": {},
    }

    comparison_visualize_samples = max(0, min(args.comparison_visualize_samples, 3))
    all_results["comparison_visualize_samples"] = comparison_visualize_samples

    python_exe = sys.executable
    train_script = str(ROOT / "tools/train_week1.py")
    eval_script = str(ROOT / "tools/evaluate_week1.py")

    for exp in experiments:
        exp_name = exp["name"]
        exp_overrides = exp.get("overrides", {})
        print(f"\n[Experiment] {exp_name}")

        exp_record: Dict[str, Any] = {
            "status": "running",
            "overrides": exp_overrides,
            "splits": {},
        }
        all_results["experiments"][exp_name] = exp_record

        try:
            cfg = build_experiment_config(
                base_cfg=base_cfg,
                exp_name=exp_name,
                exp_overrides=exp_overrides,
                output_root=output_root,
                epochs_override=args.epochs_override,
            )

            exp_config_path = generated_cfg_dir / f"{exp_name}.yaml"
            save_yaml(exp_config_path, cfg)

            run_command(
                [python_exe, train_script, "--config", str(exp_config_path)],
                cwd=ROOT,
            )

            exp_output_dir = Path(cfg["training"]["output_dir"])
            checkpoint_path = exp_output_dir / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            for split in args.evaluate_splits:
                split_results_dir = exp_output_dir / "eval" / split
                split_visualize_samples = comparison_visualize_samples if split == args.comparison_split else 0
                run_command(
                    [
                        python_exe,
                        eval_script,
                        "--config",
                        str(exp_config_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--split",
                        split,
                        "--results-dir",
                        str(split_results_dir),
                        "--visualize-samples",
                        str(split_visualize_samples),
                        "--tiny-area-threshold",
                        str(args.tiny_area_threshold),
                    ],
                    cwd=ROOT,
                )

                metrics = read_metrics(split_results_dir / "metrics.json")
                exp_record["splits"][split] = metrics

            exp_record["status"] = "completed"

        except Exception as exc:
            exp_record["status"] = "failed"
            exp_record["error"] = str(exc)
            print(f"[Error] Experiment '{exp_name}' failed: {exc}")
            if not args.continue_on_error:
                update_summary_fields(all_results, primary_split=args.comparison_split)
                save_json(output_root / "summary.json", all_results)
                raise

        update_summary_fields(all_results, primary_split=args.comparison_split)
        save_json(output_root / "summary.json", all_results)

    print("\n[Done] Baseline optimization sweep finished.")
    print(f"[Done] Summary file: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
