from __future__ import annotations

from typing import Dict

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


class DamageSegmentor(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_channels=in_channels, base_channels=base_channels)
        self.segmentation_head = SegmentationHead(in_channels=self.backbone.out_channels, num_classes=num_classes)

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
        ce = F.cross_entropy(logits, targets)
        dice = soft_dice_loss(logits, targets)
        total = ce + dice
        return {
            "loss_total": total,
            "loss_ce": ce,
            "loss_dice": dice,
        }
