import os
import json
import shutil
import argparse
import cv2


def find_file(search_dirs, sample_id):
    extensions = [".png", ".jpg", ".jpeg"]
    for folder in search_dirs:
        for ext in extensions:
            path = os.path.join(folder, f"{sample_id}{ext}")
            if os.path.exists(path):
                return path
    return None


def mask_to_polygons(mask):
    mask = (mask > 0).astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue

        cnt = cnt.squeeze()
        if len(cnt.shape) != 2 or cnt.shape[0] < 3:
            continue

        polygon = cnt.tolist()
        polygons.append(polygon)

    return polygons


def normalize_polygon(polygon, width, height):
    coords = []
    for x, y in polygon:
        coords.append(x / width)
        coords.append(y / height)
    return coords


def process_split(split_name, split_json_path, image_search_dirs, mask_search_dirs, output_dir):
    print(f"\nProcessing {split_name}...")

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    sample_ids = split_data["items"]

    out_img_dir = os.path.join(output_dir, "images", split_name)
    out_lbl_dir = os.path.join(output_dir, "labels", split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    processed = 0
    skipped = 0

    for sample_id in sample_ids:
        img_path = find_file(image_search_dirs, sample_id)
        mask_path = find_file(mask_search_dirs, sample_id)

        if img_path is None or mask_path is None:
            print(f"Skipping {sample_id} (missing file)")
            skipped += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping {sample_id} (mask unreadable)")
            skipped += 1
            continue

        h, w = mask.shape
        polygons = mask_to_polygons(mask)

        shutil.copy(img_path, os.path.join(out_img_dir, f"{sample_id}.png"))

        label_path = os.path.join(out_lbl_dir, f"{sample_id}.txt")
        with open(label_path, "w") as f:
            for polygon in polygons:
                normalized = normalize_polygon(polygon, w, h)
                if len(normalized) < 6:
                    continue
                line = "0 " + " ".join(f"{v:.6f}" for v in normalized)
                f.write(line + "\n")

        processed += 1

    print(f"{split_name} done. processed={processed}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sod-root", required=True, help="Path to CarDD_SOD root")
    parser.add_argument("--splits-dir", required=True, help="Path to data/splits")
    parser.add_argument("--output-dir", default="data/yolo_seg")
    args = parser.parse_args()

    image_search_dirs = [
        os.path.join(args.sod_root, "CarDD-TR", "CarDD-TR-Image"),
        os.path.join(args.sod_root, "CarDD-TE", "CarDD-TE-Image"),
    ]

    mask_search_dirs = [
        os.path.join(args.sod_root, "CarDD-TR", "CarDD-TR-Mask"),
        os.path.join(args.sod_root, "CarDD-TE", "CarDD-TE-Mask"),
    ]

    process_split(
        "train",
        os.path.join(args.splits_dir, "train.json"),
        image_search_dirs,
        mask_search_dirs,
        args.output_dir,
    )
    process_split(
        "val",
        os.path.join(args.splits_dir, "val.json"),
        image_search_dirs,
        mask_search_dirs,
        args.output_dir,
    )
    process_split(
        "test",
        os.path.join(args.splits_dir, "test.json"),
        image_search_dirs,
        mask_search_dirs,
        args.output_dir,
    )


if __name__ == "__main__":
    main()