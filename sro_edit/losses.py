from __future__ import annotations

import torch
import torch.nn.functional as F


def roi_module_loss(
    logits: torch.Tensor,
    soft_occupancy: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    lambda_size: float = 0.10,
    lambda_recall: float = 1.00,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper Stage-A objective: soft-target focal BCE + size + soft recall."""
    if logits.shape != soft_occupancy.shape:
        raise ValueError(f"logits/target shape mismatch: {logits.shape} vs {soft_occupancy.shape}")
    target = soft_occupancy.to(device=logits.device, dtype=logits.dtype)
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError("soft_occupancy must lie in [0,1]")

    probability = torch.sigmoid(logits)
    p_t = target * probability + (1.0 - target) * (1.0 - probability)
    alpha_t = target * float(alpha) + (1.0 - target) * (1.0 - float(alpha))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    focal = (alpha_t * (1.0 - p_t).pow(float(gamma)) * bce).mean()
    size = probability.mean()
    recall = 1.0 - ((probability * target).sum() + eps) / (target.sum() + eps)
    total = focal + float(lambda_size) * size + float(lambda_recall) * recall
    return total, {"focal": focal, "size": size, "recall": recall}


def binary_dice_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sigmoid Dice plus BCE, the segmentation term used in Stages B1 and B2."""
    if logits.shape != target.shape or logits.ndim < 3:
        raise ValueError(f"expected matching (B,C,spatial...) tensors, got {logits.shape}, {target.shape}")
    y = target.to(device=logits.device, dtype=logits.dtype)
    p = torch.sigmoid(logits)
    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (p * y).sum(dim=spatial_dims)
    denominator = p.sum(dim=spatial_dims) + y.sum(dim=spatial_dims)
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    return dice.mean() + bce


def partition_balanced_edit_loss(
    refined_logits: torch.Tensor,
    base_logits: torch.Tensor,
    target: torch.Tensor,
    editable_roi: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper B1 loss over editable base-errors and their preservation complement."""
    if refined_logits.shape != base_logits.shape or refined_logits.shape != target.shape:
        raise ValueError("refined_logits, base_logits, and target must have identical shapes")
    if editable_roi.ndim != refined_logits.ndim:
        raise ValueError("editable_roi must have the same rank as logits")
    if editable_roi.shape[0] != refined_logits.shape[0] or editable_roi.shape[2:] != refined_logits.shape[2:]:
        raise ValueError("editable_roi must share batch and spatial dimensions with logits")
    if editable_roi.shape[1] not in (1, refined_logits.shape[1]):
        raise ValueError("editable_roi channels must be 1 or match logits")

    base_probability = torch.sigmoid(base_logits.detach())
    base_prediction = base_probability >= 0.5
    y = target.to(device=refined_logits.device, dtype=refined_logits.dtype)
    gt = y >= 0.5
    roi = editable_roi.to(device=refined_logits.device) > 0.5
    if roi.shape != refined_logits.shape:
        roi = roi.expand_as(refined_logits)

    edit_error = roi & (base_prediction != gt)
    preserve = ~edit_error
    keep_map = F.binary_cross_entropy_with_logits(
        refined_logits, base_probability, reduction="none"
    )
    correct_map = F.binary_cross_entropy_with_logits(refined_logits, y, reduction="none")
    dims = tuple(range(2, refined_logits.ndim))

    keep_count = preserve.sum(dim=dims)
    edit_count = edit_error.sum(dim=dims)
    keep_active = keep_count > 0
    edit_active = edit_count > 0
    keep_mean = (keep_map * preserve).sum(dim=dims) / keep_count.clamp_min(1)
    edit_mean = (correct_map * edit_error).sum(dim=dims) / edit_count.clamp_min(1)
    active_count = keep_active.to(refined_logits.dtype) + edit_active.to(refined_logits.dtype)
    per_channel = (keep_mean + edit_mean) / active_count.clamp_min(1.0)
    valid_channel = active_count > 0
    loss = per_channel[valid_channel].mean() if bool(valid_channel.any()) else refined_logits.sum() * 0.0
    return loss, {
        "preserve": keep_mean[keep_active].mean() if bool(keep_active.any()) else loss * 0.0,
        "edit": edit_mean[edit_active].mean() if bool(edit_active.any()) else loss * 0.0,
        "editable_error_mask": edit_error,
    }


def editable_region_consistency_loss(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    editable_roi: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Paper B2 distillation loss, averaged only over editable voxels."""
    if student_logits.shape != teacher_probability.shape:
        raise ValueError("student_logits and teacher_probability must have identical shapes")
    if editable_roi.ndim != student_logits.ndim:
        raise ValueError("editable_roi must have the same rank as student_logits")
    if editable_roi.shape[0] != student_logits.shape[0] or editable_roi.shape[2:] != student_logits.shape[2:]:
        raise ValueError("editable_roi must share batch and spatial dimensions with logits")
    if editable_roi.shape[1] not in (1, student_logits.shape[1]):
        raise ValueError("editable_roi channels must be 1 or match logits")

    roi = editable_roi.to(device=student_logits.device) > 0.5
    if roi.shape != student_logits.shape:
        roi = roi.expand_as(student_logits)
    teacher = teacher_probability.detach().to(device=student_logits.device, dtype=student_logits.dtype)
    if bool(((teacher < 0) | (teacher > 1)).any()):
        raise ValueError("teacher_probability must lie in [0,1]")
    loss_map = F.binary_cross_entropy_with_logits(student_logits, teacher, reduction="none")
    dims = tuple(range(2, student_logits.ndim))
    count = roi.sum(dim=dims).to(dtype=loss_map.dtype)
    active = count > 0
    per_channel = (loss_map * roi).sum(dim=dims) / count.clamp_min(eps)
    return per_channel[active].mean() if bool(active.any()) else student_logits.sum() * 0.0


def stage_b1_loss(
    base_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    target: torch.Tensor,
    editable_roi: torch.Tensor,
    lambda_edit: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """B1 objective; the caller freezes the backbone and trains the refiner."""
    segmentation = binary_dice_bce_loss(refined_logits, target)
    edit, parts = partition_balanced_edit_loss(refined_logits, base_logits, target, editable_roi)
    total = segmentation + float(lambda_edit) * edit
    return total, {"segmentation": segmentation, "edit_preserve": edit, **parts}


def stage_b2_loss(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    target: torch.Tensor,
    editable_roi: torch.Tensor,
    lambda_ft: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """B2 objective; the caller freezes the refiner and selectively trains the backbone."""
    segmentation = binary_dice_bce_loss(student_logits, target)
    consistency = editable_region_consistency_loss(student_logits, teacher_probability, editable_roi)
    total = segmentation + float(lambda_ft) * consistency
    return total, {"segmentation": segmentation, "consistency": consistency}
