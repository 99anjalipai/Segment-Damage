from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.unet import UNetBackbone
from models.task_heads.dent_classification_head import DentClassificationHead
from models.task_heads.feature_projector import FeatureProjectionNetwork
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


def tiny_object_pixel_contrastive_loss(
    projected_features: torch.Tensor,
    targets: torch.Tensor,
    tiny_area_threshold_pixels: int = 2048,
    foreground_class_index: int = 1,
    temperature: float = 0.1,
    max_tiny_samples: int = 1024,
    max_other_samples: int = 4096,
    min_tiny_pixels: int = 16,
) -> torch.Tensor:
    b, c, h, w = projected_features.shape

    if targets.shape[-2:] != (h, w):
        targets = F.interpolate(targets.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()

    if b == 0:
        return projected_features.new_zeros(())

    if c <= 0:
        return projected_features.new_zeros(())

    if tiny_area_threshold_pixels < 1:
        tiny_area_threshold_pixels = 1

    if int(foreground_class_index) >= 0:
        fg_mask = targets == int(foreground_class_index)
    else:
        fg_mask = targets > 0

    fg_pixels_per_image = fg_mask.flatten(1).sum(dim=1)
    tiny_image_mask = (fg_pixels_per_image > 0) & (fg_pixels_per_image <= int(tiny_area_threshold_pixels))
    if not tiny_image_mask.any():
        return projected_features.new_zeros(())

    tiny_fg_mask = fg_mask & tiny_image_mask.view(-1, 1, 1)
    tiny_pixel_count = int(tiny_fg_mask.sum().item())
    if tiny_pixel_count < max(2, int(min_tiny_pixels)):
        return projected_features.new_zeros(())

    features = projected_features.permute(0, 2, 3, 1).reshape(-1, c)
    features = F.normalize(features, p=2, dim=1)

    tiny_indices = tiny_fg_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    other_indices = (~tiny_fg_mask).reshape(-1).nonzero(as_tuple=False).squeeze(1)

    max_tiny = max(2, int(max_tiny_samples))
    if tiny_indices.numel() > max_tiny:
        perm = torch.randperm(tiny_indices.numel(), device=tiny_indices.device)[:max_tiny]
        tiny_indices = tiny_indices[perm]

    max_other = max(1, int(max_other_samples))
    if other_indices.numel() > max_other:
        perm = torch.randperm(other_indices.numel(), device=other_indices.device)[:max_other]
        other_indices = other_indices[perm]

    if tiny_indices.numel() < 2 or other_indices.numel() < 1:
        return projected_features.new_zeros(())

    tiny_embeddings = features[tiny_indices]
    other_embeddings = features[other_indices]

    temp = max(float(temperature), 1e-6)
    sim_tiny = tiny_embeddings @ tiny_embeddings.T / temp
    sim_other = tiny_embeddings @ other_embeddings.T / temp

    eye_mask = torch.eye(sim_tiny.shape[0], device=sim_tiny.device, dtype=torch.bool)
    sim_tiny = sim_tiny.masked_fill(eye_mask, float("-inf"))

    if sim_tiny.shape[1] <= 1:
        return projected_features.new_zeros(())

    log_num = torch.logsumexp(sim_tiny, dim=1)
    sim_den = torch.cat([sim_tiny, sim_other], dim=1)
    log_den = torch.logsumexp(sim_den, dim=1)
    return -(log_num - log_den).mean()


def class_aware_embedding_contrastive_loss(
    embeddings: torch.Tensor,
    cls_targets: torch.Tensor | None = None,
    cls_targets_multi: torch.Tensor | None = None,
    temperature: float = 0.1,
    always_on_pairwise: bool = True,
) -> torch.Tensor:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return embeddings.new_zeros(())

    emb = F.normalize(embeddings, p=2, dim=1)
    b = emb.shape[0]
    temp = max(float(temperature), 1e-6)

    sim = emb @ emb.T / temp
    eye = torch.eye(b, device=emb.device, dtype=torch.bool)

    if cls_targets_multi is not None:
        labels_multi = (cls_targets_multi > 0.5).float()
        if labels_multi.ndim != 2 or labels_multi.shape[0] != b:
            return embeddings.new_zeros(())
        positive_mask = (labels_multi @ labels_multi.T) > 0.0
    elif cls_targets is not None:
        labels = cls_targets.view(-1)
        if labels.shape[0] != b:
            return embeddings.new_zeros(())
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    else:
        return embeddings.new_zeros(())

    positive_mask = positive_mask & (~eye)
    valid_anchor_mask = positive_mask.any(dim=1)
    if not valid_anchor_mask.any() and not bool(always_on_pairwise):
        return embeddings.new_zeros(())

    losses = []
    for i in torch.where(valid_anchor_mask)[0]:
        pos_logits = sim[i][positive_mask[i]]
        all_logits = sim[i][~eye[i]]
        if pos_logits.numel() == 0 or all_logits.numel() == 0:
            continue
        losses.append(-(torch.logsumexp(pos_logits, dim=0) - torch.logsumexp(all_logits, dim=0)))

    if losses:
        return torch.stack(losses).mean()

    # Fallback: pairwise similarity regression keeps contrastive pressure on every batch.
    cos = torch.clamp(emb @ emb.T, min=-1.0, max=1.0)
    pred_sim01 = (cos + 1.0) * 0.5

    if cls_targets_multi is not None:
        labels_multi = (cls_targets_multi > 0.5).float()
        inter = labels_multi @ labels_multi.T
        sums = labels_multi.sum(dim=1, keepdim=True)
        union = sums + sums.T - inter
        target_sim = torch.where(union > 0.0, inter / torch.clamp(union, min=1e-6), torch.zeros_like(union))
    elif cls_targets is not None:
        labels = cls_targets.view(-1)
        target_sim = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    else:
        return embeddings.new_zeros(())

    pair_mask = ~eye
    return F.binary_cross_entropy(pred_sim01[pair_mask], target_sim[pair_mask])


class DamageSegmentor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_projector_config: Dict | None = None,
        dent_classification_config: Dict | None = None,
        loss_config: Dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_channels=in_channels, base_channels=base_channels)
        projector_cfg = feature_projector_config or {}
        projector_enabled = bool(projector_cfg.get("enabled", False))

        if projector_enabled:
            projector_out_channels = int(projector_cfg.get("out_channels", self.backbone.out_channels))
            self.feature_projector = FeatureProjectionNetwork(
                in_channels=self.backbone.out_channels,
                out_channels=projector_out_channels,
                hidden_channels=projector_cfg.get("hidden_channels"),
                num_layers=int(projector_cfg.get("num_layers", 2)),
                dropout=float(projector_cfg.get("dropout", 0.0)),
                use_residual=bool(projector_cfg.get("use_residual", True)),
            )
            seg_head_in_channels = projector_out_channels
        else:
            self.feature_projector = nn.Identity()
            seg_head_in_channels = self.backbone.out_channels

        self.segmentation_head = SegmentationHead(in_channels=seg_head_in_channels, num_classes=num_classes)
        dent_cls_cfg = dent_classification_config or {}
        self.dent_classification_enabled = bool(dent_cls_cfg.get("enabled", False))
        if self.dent_classification_enabled:
            dent_num_classes = int(dent_cls_cfg.get("num_classes", 2))
            self.dent_classification_head = DentClassificationHead(
                in_channels=seg_head_in_channels,
                num_classes=dent_num_classes,
                hidden_channels=dent_cls_cfg.get("hidden_channels"),
                dropout=float(dent_cls_cfg.get("dropout", 0.0)),
            )
        else:
            self.dent_classification_head = None
        self.loss_config = loss_config or {}

    def _loss_weights(self) -> Tuple[float, float, float, float, float, float, float]:
        weights = self.loss_config.get("weights", {})
        ce_w = float(weights.get("ce", 1.0))
        dice_w = float(weights.get("dice", 1.0))
        focal_w = float(weights.get("focal", 1.0))
        grad_w = float(weights.get("grad", 1.0))
        contrastive_w = float(weights.get("contrastive", 1.0))
        cls_w = float(weights.get("cls", 1.0))
        cls_contrastive_w = float(weights.get("cls_contrastive", 1.0))
        return ce_w, dice_w, focal_w, grad_w, contrastive_w, cls_w, cls_contrastive_w

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features, pyramid = self.backbone(images)
        projected_features = self.feature_projector(features)
        logits = self.segmentation_head(projected_features)
        cls_logits = None
        cls_embedding = None
        if self.dent_classification_enabled and self.dent_classification_head is not None:
            cls_embedding = F.adaptive_avg_pool2d(projected_features, output_size=1).flatten(1)
            cls_logits = self.dent_classification_head(projected_features)

        # Week 2 and Week 3 hooks:
        # - projection embeddings from features/pyramid
        # - tiny-object contrastive module
        # - gradient boundary supervision branch
        return {
            "logits": logits,
            "cls_logits": cls_logits,
            "cls_embedding": cls_embedding,
            "features": features,
            "projected_features": projected_features,
            "pyramid": pyramid,
        }

    def compute_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        projected_features: torch.Tensor | None = None,
        cls_logits: torch.Tensor | None = None,
        cls_targets: torch.Tensor | None = None,
        cls_targets_multi: torch.Tensor | None = None,
        cls_embedding: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        loss_name = str(self.loss_config.get("name", "ce_dice")).strip().lower()
        ce_w, dice_w, focal_w, grad_w, contrastive_w, cls_w, cls_contrastive_w = self._loss_weights()
        focal_gamma = float(self.loss_config.get("focal_gamma", 2.0))
        focal_alpha = self.loss_config.get("focal_alpha")
        if focal_alpha is not None:
            focal_alpha = float(focal_alpha)
        grad_class_index = int(self.loss_config.get("gradient_foreground_class", 1))
        use_gradient = bool(self.loss_config.get("use_gradient", False))
        contrastive_cfg = self.loss_config.get("contrastive", {})
        use_contrastive = bool(
            contrastive_cfg.get("enabled", self.loss_config.get("use_contrastive", False))
        )
        cls_cfg = self.loss_config.get("classification", {})
        use_cls = bool(cls_cfg.get("enabled", False))
        cls_multilabel = bool(cls_cfg.get("multilabel", False))
        cls_contrastive_cfg = cls_cfg.get("class_contrastive", {})
        use_cls_contrastive = bool(cls_contrastive_cfg.get("enabled", False))

        ce = F.cross_entropy(logits, targets)
        dice = soft_dice_loss(logits, targets)
        focal = focal_loss(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
        grad = gradient_boundary_loss(
            logits,
            targets,
            foreground_class_index=grad_class_index,
        )
        contrastive = logits.new_zeros(())
        if use_contrastive and projected_features is not None:
            contrastive = tiny_object_pixel_contrastive_loss(
                projected_features=projected_features,
                targets=targets,
                tiny_area_threshold_pixels=int(contrastive_cfg.get("tiny_area_threshold_pixels", 2048)),
                foreground_class_index=int(contrastive_cfg.get("foreground_class_index", 1)),
                temperature=float(contrastive_cfg.get("temperature", 0.1)),
                max_tiny_samples=int(contrastive_cfg.get("max_tiny_samples", 1024)),
                max_other_samples=int(contrastive_cfg.get("max_other_samples", 4096)),
                min_tiny_pixels=int(contrastive_cfg.get("min_tiny_pixels", 16)),
            )

        cls_loss = logits.new_zeros(())
        if use_cls and cls_logits is not None:
            if cls_multilabel and cls_targets_multi is not None:
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets_multi.float())
            elif cls_targets is not None:
                cls_loss = F.cross_entropy(cls_logits, cls_targets.long())

        cls_contrastive_loss = logits.new_zeros(())
        if use_cls_contrastive and cls_embedding is not None:
            cls_contrastive_loss = class_aware_embedding_contrastive_loss(
                embeddings=cls_embedding,
                cls_targets=cls_targets,
                cls_targets_multi=cls_targets_multi,
                temperature=float(cls_contrastive_cfg.get("temperature", 0.1)),
                always_on_pairwise=bool(cls_contrastive_cfg.get("always_on_pairwise", True)),
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
        if use_contrastive:
            total = total + contrastive_w * contrastive
        if use_cls:
            total = total + cls_w * cls_loss
        if use_cls_contrastive:
            total = total + cls_contrastive_w * cls_contrastive_loss

        return {
            "loss_total": total,
            "loss_ce": ce,
            "loss_dice": dice,
            "loss_focal": focal,
            "loss_grad": grad,
            "loss_contrastive": contrastive,
            "loss_cls": cls_loss,
            "loss_cls_contrastive": cls_contrastive_loss,
        }