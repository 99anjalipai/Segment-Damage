from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.unet import UNetBackbone
from models.task_heads.segmentation_head import SegmentationHead


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    denominator = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal_factor = (1.0 - pt).clamp(min=0.0, max=1.0) ** gamma

    if alpha is None:
        alpha_factor = 1.0
    else:
        alpha_factor = torch.where(targets > 0, alpha, 1.0 - alpha).float()

    loss = alpha_factor * focal_factor * ce
    return loss.mean() + 0.0 * eps


def gradient_boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    foreground_class_index: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    fg_index = max(0, min(foreground_class_index, logits.shape[1] - 1))
    pred_fg = probs[:, fg_index : fg_index + 1]

    # For binary segmentation use class index; for multi-class treat any non-background as foreground.
    if logits.shape[1] == 2:
        target_fg = (targets == fg_index).float().unsqueeze(1)
    else:
        target_fg = (targets > 0).float().unsqueeze(1)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=logits.device,
        dtype=logits.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=logits.device,
        dtype=logits.dtype,
    ).view(1, 1, 3, 3)

    pred_gx = F.conv2d(pred_fg, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_fg, sobel_y, padding=1)
    tgt_gx = F.conv2d(target_fg, sobel_x, padding=1)
    tgt_gy = F.conv2d(target_fg, sobel_y, padding=1)

    pred_grad = torch.sqrt(pred_gx * pred_gx + pred_gy * pred_gy + eps)
    tgt_grad = torch.sqrt(tgt_gx * tgt_gx + tgt_gy * tgt_gy + eps)
    return F.l1_loss(pred_grad, tgt_grad)


class DamageSegmentor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 32,
        loss_config: Dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_channels=in_channels, base_channels=base_channels)
        self.segmentation_head = SegmentationHead(in_channels=self.backbone.out_channels, num_classes=num_classes)
        self.loss_config = loss_config or {}

    def _loss_weights(self) -> Tuple[float, float, float, float]:
        weights = self.loss_config.get("weights", {})
        ce_w = float(weights.get("ce", 1.0))
        dice_w = float(weights.get("dice", 1.0))
        focal_w = float(weights.get("focal", 1.0))
        grad_w = float(weights.get("grad", 1.0))
        return ce_w, dice_w, focal_w, grad_w

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features, pyramid = self.backbone(images)
        logits = self.segmentation_head(features)

        # Week 2 and Week 3 hooks:
        # - projection embeddings from features/pyramid
        # - tiny-object contrastive module
        # - gradient boundary supervision branch
        return {
            "logits": logits,
            "features": features,
            "pyramid": pyramid,
        }

    def compute_losses(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_name = str(self.loss_config.get("name", "ce_dice")).strip().lower()
        ce_w, dice_w, focal_w, grad_w = self._loss_weights()
        focal_gamma = float(self.loss_config.get("focal_gamma", 2.0))
        focal_alpha = self.loss_config.get("focal_alpha")
        if focal_alpha is not None:
            focal_alpha = float(focal_alpha)
        grad_class_index = int(self.loss_config.get("gradient_foreground_class", 1))
        use_gradient = bool(self.loss_config.get("use_gradient", False))

        ce = F.cross_entropy(logits, targets)
        dice = soft_dice_loss(logits, targets)
        focal = focal_loss(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
        grad = gradient_boundary_loss(
            logits,
            targets,
            foreground_class_index=grad_class_index,
        )

        if loss_name in {"ce", "cross_entropy", "crossentropy"}:
            total = ce_w * ce
        elif loss_name == "dice":
            total = dice_w * dice
        elif loss_name == "focal":
            total = focal_w * focal
        elif loss_name in {"grad", "gradient", "gradient_boundary"}:
            total = grad_w * grad
        elif loss_name in {"ce_dice", "dice_ce"}:
            total = ce_w * ce + dice_w * dice
        elif loss_name in {"ce_focal", "focal_ce"}:
            total = ce_w * ce + focal_w * focal
        elif loss_name in {"dice_focal", "focal_dice"}:
            total = dice_w * dice + focal_w * focal
        elif loss_name in {"ce_dice_focal", "focal_ce_dice"}:
            total = ce_w * ce + dice_w * dice + focal_w * focal
        else:
            raise ValueError(
                "Unsupported loss name '"
                f"{loss_name}'. Use one of: ce, dice, focal, grad, ce_dice, ce_focal, dice_focal, ce_dice_focal"
            )

        if use_gradient and loss_name not in {"grad", "gradient", "gradient_boundary"}:
            total = total + grad_w * grad

        return {
            "loss_total": total,
            "loss_ce": ce,
            "loss_dice": dice,
            "loss_focal": focal,
            "loss_grad": grad,
        }
