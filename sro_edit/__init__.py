"""Paper-aligned core components for SRO-Edit."""

from .losses import (
    binary_dice_bce_loss,
    editable_region_consistency_loss,
    partition_balanced_edit_loss,
    roi_module_loss,
    stage_b1_loss,
    stage_b2_loss,
)
from .roi import (
    apply_roi_state_gate,
    build_editable_roi,
    encode_click_spheres,
    make_refiner_input,
    normalized_round_weight,
    supervoxel_occupancy_targets,
)

__all__ = [
    "apply_roi_state_gate",
    "binary_dice_bce_loss",
    "build_editable_roi",
    "editable_region_consistency_loss",
    "encode_click_spheres",
    "make_refiner_input",
    "normalized_round_weight",
    "partition_balanced_edit_loss",
    "roi_module_loss",
    "stage_b1_loss",
    "stage_b2_loss",
    "supervoxel_occupancy_targets",
]
