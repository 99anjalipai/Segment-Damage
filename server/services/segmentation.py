import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import yaml
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

# Add the repository root to sys.path to allow importing from `models`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DamageSegmentor

# Global model variables
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _load_model(model_name="fpn_ce_dice_focal_grad_contrastive_tuned_v2"):
    """
    Loads the segmentation model and weights for the given model_name (folder in outputs/ and yaml in configs/).
    """
    config_path = ROOT / "configs" / f"{model_name}.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    loss_cfg = cfg.get("training", {}).get("loss", {})
    model = DamageSegmentor(
        num_classes=model_cfg["num_classes"],
        in_channels=model_cfg.get("in_channels", 3),
        base_channels=model_cfg.get("base_channels", 32),
        feature_projector_config=model_cfg.get("feature_projector", {}),
        dent_classification_config=model_cfg.get("dent_classification", {}),
        loss_config=loss_cfg,
    )
    weights_path = ROOT / "outputs" / model_name / "best.pt"
    ckpt = torch.load(weights_path, map_location=_device)
    # Load state dict
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(_device)
    model.eval()
    return model


def segment_damage(image_array, model_name="fpn_ce_dice_focal_grad_contrastive_tuned_v2"):
    """
    Runs actual PyTorch Unet inference using the selected model.
    """
    model = _load_model(model_name)
    # Process PIL image from numpy array
    image = Image.fromarray(image_array)
    original_size = image.size  # (width, height)
    # Preprocess
    image_resized = TF.resize(image, [512, 512], InterpolationMode.BILINEAR)
    img_tensor = TF.to_tensor(image_resized)
    img_tensor = TF.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).unsqueeze(0).to(_device)
    # Inference
    with torch.no_grad():
        out = model(img_tensor)
        logits = out["logits"]
        # Taking argmax assuming cross entropy / focal architecture with > 1 classes
        if logits.shape[1] > 1:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = (torch.sigmoid(logits) > 0.5).int()
        mask = preds.squeeze(0).cpu().numpy().astype(np.uint8)
    # Resize mask back to original image size
    mask_img = Image.fromarray(mask)
    # Using nearest neighbor interpolation for the mask resizing
    mask_img = mask_img.resize(original_size, resample=Image.NEAREST)
    return np.array(mask_img)

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    """
    Overlays a binary mask onto an image.
    """
    img_array = np.array(image).copy()
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    overlay = img_array.copy()
    overlay[mask == 1] = color
    
    output = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)
    return output