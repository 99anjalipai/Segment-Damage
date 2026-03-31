import os
import json
import shutil
import argparse
import cv2
import numpy as np


def find_file(base_dirs, subfolder, sample_id):
    """
    Search for file in train/test folders.
    """
    for base in base_dirs:
        path = os.path.join(base, subfolder, f"{sample_id}.png")
        if os.path.exists(path):
            return path
    return None


def mask_to_polygons(mask):
    """
    Convert binary mask to list of polygons (contours).
    """
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        polygon = cnt.squeeze().tolist()
        if len(polygon) < 3:
            continue
        polygons.append(polygon)

    return polygons


def normalize_polygon(polygon, width, height):
    """
    Normalize polygon coordinates to [0,1].
    """
    normalized = []
    for x, y in polygon:
        normalized.append(x / width)
        normalized.append(y / height)
    return normalized


def process_split(split_name, split_json_path, image_dirs, mask_dirs, output_dir):
    print(f"\nProcessing {split_name}...")

    with open(split_json_path, "r") as f:
        data = json.load(f)

    ids = data["items"]

    img_out_dir = os.path.join(output_dir, "images", split_name)
    lbl_out_dir = os.path.join(output_dir, "labels", split_name)

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for sample_id in ids:
        img_path = find_file(image_dirs, "image", sample_id)
        mask_path = find_file(mask_dirs, "mask", sample_id)

        if img_path is None or mask_path is None:
            print(f"Skipping {sample_id} (missing file)")
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w = mask.shape

        polygons = mask_to_polygons(mask)

        # Copy image
        shutil.copy(img_path, os.path.join(img_out_dir, f"{sample_id}.png"))

        label_path = os.path.join(lbl_out_dir, f"{sample_id}.txt")

        with open(label_path, "w") as f:
            for polygon in polygons:
                normalized = normalize_polygon(polygon, w, h)

                if len(normalized) < 6:
                    continue

                line = "0 " + " ".join([f"{p:.6f}" for p in normalized])
                f.write(line + "\n")

    print(f"{split_name} done.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sod-train", required=True, help="Path to cardd_sod/train")
    parser.add_argument("--sod-test", required=True, help="Path to cardd_sod/test")
    parser.add_argument("--splits-dir", required=True, help="Path to data/splits")
    parser.add_argument("--output-dir", default="data/yolo_seg")

    args = parser.parse_args()

    image_dirs = [
        args.sod_train,
        args.sod_test
    ]

    mask_dirs = [
        args.sod_train,
        args.sod_test
    ]

    process_split("train",
                  os.path.join(args.splits_dir, "train.json"),
                  image_dirs, mask_dirs, args.output_dir)

    process_split("val",
                  os.path.join(args.splits_dir, "val.json"),
                  image_dirs, mask_dirs, args.output_dir)

    process_split("test",
                  os.path.join(args.splits_dir, "test.json"),
                  image_dirs, mask_dirs, args.output_dir)


if __name__ == "__main__":
    main()