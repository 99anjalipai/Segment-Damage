import argparse
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"]["pretrained_weights"])

    model.train(
        data=cfg["dataset"]["yolo_data_yaml"],
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["model"]["imgsz"],
        batch=cfg["training"]["batch"],
        device=cfg["training"]["device"],
        project=cfg["training"]["project"],
        name=cfg["training"]["name"],
        patience=cfg["training"]["patience"],
        workers=cfg["training"]["workers"],
        seed=cfg["training"]["seed"],
    )


if __name__ == "__main__":
    main()