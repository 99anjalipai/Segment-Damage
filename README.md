# Segment-Damage (Week 1 Setup)

This repository now includes a Week 1 baseline architecture for vehicle damage segmentation on CarDD.

## What is implemented

- Dataset split creation utility (train/val/test)
- CarDD image-mask dataloader with standard augmentation
- Modular baseline segmentation architecture
  - U-Net backbone
  - Segmentation head
  - CE + Dice training loss
- Training script with checkpointing (`best.pt`, `last.pt`)
- Evaluation script with mIoU and tiny-damage `DET_l` proxy metric

## Project layout

- `data/cardd_dataset.py`: dataset class and dataloader builder
- `models/backbone/unet.py`: U-Net encoder-decoder backbone
- `models/task_heads/segmentation_head.py`: segmentation logits head
- `models/segmentor.py`: end-to-end model + losses + Week 2/3 hooks
- `tools/prepare_cardd_splits.py`: split generator
- `tools/train_week1.py`: baseline training entrypoint
- `tools/evaluate_week1.py`: baseline evaluation entrypoint
- `configs/week1_unet.yaml`: baseline config

## Expected CarDD structure

Place data as:

```
data/
  cardd/
    images/
    masks/
```

Filenames must share the same stem for pairing (example: `img_001.jpg` and `img_001.png`).

## Week 1 run steps

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create train/val/test splits:

```bash
python tools/prepare_cardd_splits.py \
  --image-dir data/cardd/images \
  --mask-dir data/cardd/masks \
  --output-dir data/splits
```

3. Train baseline:

```bash
python tools/train_week1.py --config configs/week1_unet.yaml
```

4. Evaluate baseline:

```bash
python tools/evaluate_week1.py \
  --config configs/week1_unet.yaml \
  --checkpoint outputs/week1_unet/best.pt
```

## Notes for Week 2 and Week 3

The model wrapper includes explicit extension points for:

- Pixel embedding projection layer
- Tiny-object contrastive module
- Gradient boundary supervision module

These can be integrated without changing the overall training script structure.
