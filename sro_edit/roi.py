from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def _volume3d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (D,H,W), got {array.shape}")
    if 0 in array.shape:
        raise ValueError(f"{name} must have non-empty spatial dimensions")
    return array


def supervoxel_occupancy_targets(
    supervoxels: np.ndarray,
    foreground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted supervoxel IDs and their soft foreground occupancy.

    The target for supervoxel i is |SV_i intersect foreground| / |SV_i|.
    It is the paper's soft structural target, not a hard threshold or an
    error-region target.
    """
    sv = _volume3d(supervoxels, "supervoxels").astype(np.int64, copy=False)
    fg = _volume3d(foreground, "foreground").astype(bool, copy=False)
    if sv.shape != fg.shape:
        raise ValueError(f"shape mismatch: supervoxels={sv.shape}, foreground={fg.shape}")

    ids, inverse = np.unique(sv.reshape(-1), return_inverse=True)
    counts = np.bincount(inverse, minlength=ids.size).astype(np.float64)
    foreground_counts = np.bincount(
        inverse, weights=fg.reshape(-1).astype(np.float64), minlength=ids.size
    )
    occupancy = (foreground_counts / counts).astype(np.float32)
    return ids.astype(np.int64, copy=False), occupancy


def build_editable_roi(
    supervoxels: np.ndarray,
    supervoxel_ids: Sequence[int] | np.ndarray,
    relevance_scores: Sequence[float] | np.ndarray,
    threshold: float,
    positive_clicks: Iterable[Sequence[int]] = (),
    negative_clicks: Iterable[Sequence[int]] = (),
) -> np.ndarray:
    """Threshold learned supervoxel scores and include every clicked SV.

    Click coordinates use array order (D,H,W). ``relevance_scores`` are
    sigmoid probabilities aligned one-to-one with ``supervoxel_ids``.
    """
    sv = _volume3d(supervoxels, "supervoxels").astype(np.int64, copy=False)
    ids = np.asarray(supervoxel_ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(relevance_scores, dtype=np.float32).reshape(-1)
    tau = float(threshold)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {tau}")
    if ids.size != scores.size:
        raise ValueError(f"got {ids.size} supervoxel IDs and {scores.size} scores")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("relevance_scores must be finite probabilities in [0,1]")
    actual_ids = np.unique(sv)
    if not np.array_equal(np.sort(ids), actual_ids):
        raise ValueError("supervoxel_ids must match the labels in supervoxels")

    selected: set[int] = set(int(v) for v in ids[scores >= tau])
    for point in (*positive_clicks, *negative_clicks):
        coord = tuple(int(v) for v in point)
        if len(coord) != 3 or any(v < 0 or v >= sv.shape[d] for d, v in enumerate(coord)):
            raise ValueError(f"click {coord} is outside volume shape {sv.shape}")
        selected.add(int(sv[coord]))

    if not selected:
        return np.zeros(sv.shape, dtype=bool)
    return np.isin(sv, np.fromiter(sorted(selected), dtype=np.int64))


def encode_click_spheres(
    shape: Sequence[int],
    clicks: Iterable[Sequence[int]],
    radius: int = 2,
) -> np.ndarray:
    """Rasterize clicks as binary Euclidean 3D spheres, as specified in paper."""
    spatial_shape = tuple(int(v) for v in shape)
    if len(spatial_shape) != 3 or any(v <= 0 for v in spatial_shape):
        raise ValueError(f"shape must be three positive dimensions, got {spatial_shape}")
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}")

    result = np.zeros(spatial_shape, dtype=bool)
    for raw_point in clicks:
        point = tuple(int(v) for v in raw_point)
        if len(point) != 3 or any(v < 0 or v >= spatial_shape[d] for d, v in enumerate(point)):
            raise ValueError(f"click {point} is outside volume shape {spatial_shape}")
        z, y, x = point
        z0, z1 = max(0, z - r), min(spatial_shape[0], z + r + 1)
        y0, y1 = max(0, y - r), min(spatial_shape[1], y + r + 1)
        x0, x1 = max(0, x - r), min(spatial_shape[2], x + r + 1)
        zz, yy, xx = np.ogrid[z0 - z : z1 - z, y0 - y : y1 - y, x0 - x : x1 - x]
        result[z0:z1, y0:y1, x0:x1] |= (zz * zz + yy * yy + xx * xx) <= r * r
    return result


def apply_roi_state_gate(
    current_prediction: np.ndarray,
    editable_roi: np.ndarray,
    previous_state: np.ndarray,
) -> np.ndarray:
    """Update the binary state inside the ROI and retain it outside the ROI."""
    current = _volume3d(current_prediction, "current_prediction").astype(bool, copy=False)
    roi = _volume3d(editable_roi, "editable_roi").astype(bool, copy=False)
    previous = _volume3d(previous_state, "previous_state").astype(bool, copy=False)
    if current.shape != roi.shape or current.shape != previous.shape:
        raise ValueError(
            f"shape mismatch: current={current.shape}, roi={roi.shape}, previous={previous.shape}"
        )
    return np.where(roi, current, previous)


def make_refiner_input(
    image: torch.Tensor,
    interaction: torch.Tensor,
    editable_roi: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the B1/B2 refiner inputs [X, I_k, M_edit,k, sigmoid(B_k)]."""
    tensors = (image, interaction, editable_roi, base_logits)
    if any(t.ndim != 5 for t in tensors):
        raise ValueError("all tensors must have shape (B,C,D,H,W)")
    if any(t.shape[0] != image.shape[0] or t.shape[2:] != image.shape[2:] for t in tensors[1:]):
        raise ValueError("refiner inputs must share batch and spatial dimensions")
    if editable_roi.shape[1] != 1:
        raise ValueError("editable_roi must have one channel")
    base_probability = torch.sigmoid(base_logits.detach()).to(dtype=image.dtype)
    roi = (editable_roi > 0.5).to(dtype=image.dtype)
    return torch.cat((image, interaction.to(image.dtype), roi, base_probability), dim=1)


def normalized_round_weight(round_number: int, rounds: int) -> float:
    """Paper weight w_k=2k/[K(K+1)], with one-based round_number."""
    k, total = int(round_number), int(rounds)
    if total < 1 or not 1 <= k <= total:
        raise ValueError(f"require 1 <= round_number <= rounds, got {k}/{total}")
    return (2.0 * k) / (total * (total + 1.0))
