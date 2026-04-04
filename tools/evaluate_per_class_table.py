import argparse
import json
import os

def generate_table_from_metrics(metrics_path, method_name="Ours (TinyDamage)"):
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # We use these exact names as defined in the paper text:
    categories = ["Scratch", "Dent", "Crack", "Glass Shatter", "Lamp Broken", "Tire Flat"]
    index_to_name = metrics.get('cls_index_to_name', {
        "0": "dent",
        "1": "scratch",
        "2": "crack",
        "3": "glass shatter",
        "4": "lamp broken",
        "5": "tire flat"
    })
    
    name_to_idx = {name.lower(): int(idx) for idx, name in index_to_name.items()}
    cat_names_normalized = {
        "Scratch": "scratch",
        "Dent": "dent",
        "Crack": "crack",
        "Glass Shatter": "glass shatter",
        "Lamp Broken": "lamp broken",
        "Tire Flat": "tire flat"
    }

    print("--- LaTeX Table Row Output ---")
    print(f"%-15s & %-5s & %-5s & %-5s & %-5s \\\\" % ("Category", "Mask2Former", "YOLO", "M2F", method_name))
    print("-" * 55)
    
    iou_per_class = metrics.get("IoU_per_class", [])

    for cat in categories:
        cat_lower = cat_names_normalized[cat]
        class_idx = name_to_idx.get(cat_lower)
        if class_idx is not None and class_idx < len(iou_per_class):
            score = iou_per_class[class_idx]
            print(f"%-15s & ...   & ...   & ...   & {score:.3f} \\\\" % (cat))
        else:
            print(f"%-15s & ...   & ...   & ...   & N/A \\\\" % (cat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True, help="Path to evaluation metrics.json (e.g. outputs/fpn_ce_dice_focal_grad_contrastive_tuned/eval/test/metrics.json)")
    parser.add_argument("--name", type=str, default="Ours (TinyDamage)")
    args = parser.parse_args()
    
    generate_table_from_metrics(args.metrics, args.name)
