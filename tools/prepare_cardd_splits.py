from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
MASK_SUFFIXES = (".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split json files from CarDD directories.")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_stems(path: Path, suffixes: tuple[str, ...]) -> List[str]:
    stems = []
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in suffixes:
            stems.append(item.stem)
    return sorted(stems)


def write_split(path: Path, items: List[str]) -> None:
    payload = {"count": len(items), "items": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stems = set(list_stems(image_dir, IMAGE_SUFFIXES))
    mask_stems = set(list_stems(mask_dir, MASK_SUFFIXES))
    shared = sorted(image_stems.intersection(mask_stems))

    if not shared:
        raise ValueError("No overlapping image/mask stems found.")

    rng.shuffle(shared)
    n_total = len(shared)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_items = shared[:n_train]
    val_items = shared[n_train : n_train + n_val]
    test_items = shared[n_train + n_val :]

    write_split(output_dir / "train.json", train_items)
    write_split(output_dir / "val.json", val_items)
    write_split(output_dir / "test.json", test_items)

    print(
        json.dumps(
            {
                "total": n_total,
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
