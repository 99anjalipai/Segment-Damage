from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


DEFAULT_ANNOTATIONS = [
    "CarDD_release/CarDD_COCO/annotations/instances_train2017.json",
    "CarDD_release/CarDD_COCO/annotations/instances_val2017.json",
    "CarDD_release/CarDD_COCO/annotations/instances_test2017.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-image dent class labels from CarDD COCO annotations. "
            "Output is compatible with dataset.class_labels_file."
        )
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=DEFAULT_ANNOTATIONS,
        help="One or more COCO instances*.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits/dent_class_labels.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="max_area",
        choices=["max_area", "max_count", "single_only"],
        help=(
            "How to choose one class per image when multiple categories exist: "
            "max_area=sum area winner, max_count=most instances winner, single_only=keep only single-category images."
        ),
    )
    parser.add_argument(
        "--background-index",
        type=int,
        default=0,
        help="Fallback class index when an image has no annotations.",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="multilabel",
        choices=["multilabel", "single"],
        help="Emit list-of-class-indices per image (multilabel) or one class index per image (single).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    ann_paths = [Path(p) for p in args.annotations]
    missing = [str(p) for p in ann_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing annotation files: {missing}")

    all_categories: Dict[int, str] = {}
    image_name_by_id: Dict[int, str] = {}
    category_area_by_image: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    category_count_by_image: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for ann_path in ann_paths:
        coco = load_json(ann_path)

        for cat in coco.get("categories", []):
            cat_id = int(cat["id"])
            all_categories[cat_id] = str(cat.get("name", f"cat_{cat_id}"))

        for img in coco.get("images", []):
            img_id = int(img["id"])
            file_name = str(img.get("file_name", ""))
            if file_name:
                image_name_by_id[img_id] = Path(file_name).stem

        for ann in coco.get("annotations", []):
            img_id = int(ann["image_id"])
            cat_id = int(ann["category_id"])
            area = float(ann.get("area", 0.0))
            category_area_by_image[img_id][cat_id] += area
            category_count_by_image[img_id][cat_id] += 1

    sorted_cat_ids = sorted(all_categories.keys())
    category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    index_to_name = {idx: all_categories[cat_id] for cat_id, idx in category_id_to_index.items()}

    labels: Dict[str, int | List[int]] = {}
    dropped_multi = 0
    for img_id, stem in image_name_by_id.items():
        area_map = category_area_by_image.get(img_id, {})
        count_map = category_count_by_image.get(img_id, {})

        if not area_map:
            if args.label_type == "multilabel":
                labels[stem] = []
            else:
                labels[stem] = int(args.background_index)
            continue

        present_cat_ids = list(area_map.keys())
        if args.label_type == "single" and args.mode == "single_only" and len(present_cat_ids) != 1:
            dropped_multi += 1
            continue

        if args.label_type == "multilabel":
            labels[stem] = sorted(int(category_id_to_index[cid]) for cid in present_cat_ids)
            continue

        if args.mode == "max_count":
            winner_cat = max(present_cat_ids, key=lambda cid: (count_map.get(cid, 0), area_map.get(cid, 0.0), -cid))
        else:
            winner_cat = max(present_cat_ids, key=lambda cid: (area_map.get(cid, 0.0), count_map.get(cid, 0), -cid))

        labels[stem] = int(category_id_to_index[winner_cat])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "sources": [p.as_posix() for p in ann_paths],
            "label_type": args.label_type,
            "mode": args.mode,
            "background_index": int(args.background_index),
            "num_images_with_labels": len(labels),
            "dropped_multi_category_images": int(dropped_multi),
            "category_id_to_index": {str(k): int(v) for k, v in category_id_to_index.items()},
            "index_to_name": {str(k): v for k, v in index_to_name.items()},
        },
        "labels": labels,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Info] Wrote labels for {len(labels)} images -> {output_path}")


if __name__ == "__main__":
    main()
