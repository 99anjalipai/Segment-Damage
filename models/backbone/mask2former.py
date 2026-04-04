from __future__ import annotations

"""
Mask2Former semantic segmentation backbone.

Outputs dense per-pixel class logits [B, num_classes, H, W] for compatibility with
the repo evaluator (argmax over logits -> mIoU, DET_l, etc.).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerBackbone(nn.Module):
    """
    Hugging Face Mask2Former wrapper for semantic segmentation.

    The universal segmentation model predicts per-query class logits and mask logits;
    we fuse them into dense per-class logits, then resize to the input resolution.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str,
        ignore_mismatched_sizes: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.pretrained_model_name = str(pretrained_model_name)

        # Match downstream dataset class count; HF uses `num_labels` for semantic classes.
        config = AutoConfig.from_pretrained(self.pretrained_model_name)
        config.num_labels = self.num_classes

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

    def _resize_logits(self, logits: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized float tensor [B, 3, H, W] (no PIL; caller handles normalization).

        Returns:
            Semantic segmentation logits [B, num_classes, H, W].
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected x of shape [B, 3, H, W], got {tuple(x.shape)}")

        b, _, h, w = x.shape

        outputs = self.model(pixel_values=x, return_dict=True)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # class_queries_logits: [B, Q, num_labels + 1] (last dim is "no-object")
        # masks_queries_logits: [B, Q, Hm, Wm]
        class_probs = F.softmax(class_queries_logits, dim=-1)[..., :-1]
        mask_probs = masks_queries_logits.sigmoid()

        # Dense semantic logits: sum_q P(class|q) * sigmoid(mask_q)
        sem_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        # Align channel dim with num_classes (handles head resize / padding edge cases).
        c_out = sem_logits.shape[1]
        if c_out != self.num_classes:
            if c_out > self.num_classes:
                sem_logits = sem_logits[:, : self.num_classes]
            else:
                padded = sem_logits.new_zeros((b, self.num_classes, sem_logits.shape[-2], sem_logits.shape[-1]))
                padded[:, :c_out] = sem_logits
                sem_logits = padded

        return self._resize_logits(sem_logits, (h, w))
