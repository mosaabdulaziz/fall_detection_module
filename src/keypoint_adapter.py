"""keypoint_adapter.py - Native 17-COCO-Keypoint Sanitization, Resampling & Feature Generation.

Implements:
1. sanitize_keypoints_17: forward-fill imputation and boundary clamping.
2. resample_window_25hz: uniform 25.0 Hz temporal grid resampling with gap reset.
3. prenormalize_2d: scene-relative coordinate normalization.
4. gen_ske_feat: 4-stream skeleton features with imputation-jump masking.
5. uniform_sample_and_format_gcn: 100-frame test-mode modulo tiling and M=2 formatting.
6. build_mmaction2_tensors: end-to-end preprocessing pipeline.
"""
from typing import Sequence, Tuple, Dict, Optional, Union
import numpy as np
import torch

COCO_BONE_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
    (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14)
)


def sanitize_keypoints_17(
    kp_17: np.ndarray,
    box_xyxy: np.ndarray,
    prev_kp_17: Optional[np.ndarray] = None,
    conf_kp: float = 0.3,
    score_mode: str = "raw"
) -> Tuple[np.ndarray, np.ndarray]:
    """Sanitizes raw keypoints via causal forward-fill or bounding-box center fallback.

    Returns:
        clean_kp_17: np.ndarray of shape (17, 3), strictly finite coordinates.
        is_confident: np.ndarray of shape (17,), boolean mask of true detections.
    """
    kp = np.asarray(kp_17, dtype=np.float32)
    box = np.asarray(box_xyxy, dtype=np.float32)

    is_confident = (
        np.isfinite(kp[:, 0]) &
        np.isfinite(kp[:, 1]) &
        np.isfinite(kp[:, 2]) &
        (kp[:, 2] >= conf_kp) &
        ~((kp[:, 0] == 0.0) & (kp[:, 1] == 0.0))
    )

    clean_kp = kp.copy()
    cx = float((box[0] + box[2]) / 2.0)
    cy = float((box[1] + box[3]) / 2.0)

    for i in range(17):
        if not is_confident[i]:
            if prev_kp_17 is not None and np.isfinite(prev_kp_17[i, 0]) and np.isfinite(prev_kp_17[i, 1]):
                clean_kp[i, :2] = prev_kp_17[i, :2]
                clean_kp[i, 2] = conf_kp
            else:
                clean_kp[i, 0] = cx
                clean_kp[i, 1] = cy
                clean_kp[i, 2] = conf_kp

    return clean_kp, is_confident


def resample_window_25hz(
    timestamps: Sequence[float],
    kp17_list: Sequence[np.ndarray],
    conf_mask_list: Sequence[np.ndarray],
    window_s: float = 1.16,
    num_frames: int = 30,
    min_window_s: float = 1.16,
    max_gap_s: float = 1.0,
    min_observations: int = 6
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resamples variable-FPS track observations onto a uniform temporal grid."""
    if len(timestamps) < min_observations:
        return None

    ts = np.asarray(timestamps, dtype=np.float64)
    t_end = ts[-1]
    span = t_end - ts[0]
    required_min = float(min_window_s)
    if span < (required_min - 1e-3):
        return None

    actual_win = min(float(window_s), span)
    t_win_start = t_end - actual_win

    # Check for tracking gaps exceeding max_gap_s intersecting or inside [t_win_start, t_end]
    first_in_win = int(np.searchsorted(ts, t_win_start))
    slice_start = max(0, first_in_win - 1)
    ts_win = ts[slice_start:]
    if len(ts_win) < 2 or np.any(np.diff(ts_win) > max_gap_s):
        return None

    kps = np.asarray(kp17_list, dtype=np.float32)
    masks = np.asarray(conf_mask_list, dtype=bool)

    # Strictly monotonic grid
    t_grid = np.linspace(t_end - actual_win, t_end, num_frames)

    resampled_pts = np.zeros((num_frames, 17, 3), dtype=np.float32)
    for j in range(17):
        for c in range(3):
            resampled_pts[:, j, c] = np.interp(t_grid, ts, kps[:, j, c])

    # Nearest-neighbor interpolation for boolean confidence mask
    nearest_idx = np.argmin(np.abs(ts[:, None] - t_grid[None, :]), axis=0)
    resampled_mask = masks[nearest_idx]

    return resampled_pts, resampled_mask


def prenormalize_2d(
    kp_t_17_3: np.ndarray,
    img_shape: Tuple[int, int] = (720, 1280),
    norm_mode: str = "prenormalize2d",
    compact_scale_factor: float = 1.25
) -> np.ndarray:
    """Normalizes keypoints into canonical [-1, 1] coordinate space."""
    h, w = img_shape
    h = max(float(h), 1.0)
    w = max(float(w), 1.0)

    out = np.asarray(kp_t_17_3, dtype=np.float32).copy()

    if norm_mode == "prenormalize2d":
        out[..., 0] = (out[..., 0] - w / 2.0) / (w / 2.0)
        out[..., 1] = (out[..., 1] - h / 2.0) / (h / 2.0)
    elif norm_mode == "track_compact":
        x_min = np.nanmin(out[..., 0])
        x_max = np.nanmax(out[..., 0])
        y_min = np.nanmin(out[..., 1])
        y_max = np.nanmax(out[..., 1])
        bw = max(x_max - x_min, 1.0)
        bh = max(y_max - y_min, 1.0)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        scale = max(bw, bh) * float(compact_scale_factor) / 2.0
        out[..., 0] = (out[..., 0] - cx) / scale
        out[..., 1] = (out[..., 1] - cy) / scale
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    return out


def gen_ske_feat(
    j_norm: np.ndarray,
    conf_mask: Optional[np.ndarray] = None,
    streams: Sequence[str] = ("j", "jm", "b", "bm")
) -> Dict[str, np.ndarray]:
    """Generates requested skeleton feature streams with imputation-jump masking."""
    j = np.asarray(j_norm, dtype=np.float32)
    if j.ndim == 3:
        j = j[None, ...]  # shape (1, T, 17, 3)

    _, T, V, C = j.shape
    feats: Dict[str, np.ndarray] = {}

    if "j" in streams:
        feats["j"] = j.copy()

    # Bone stream
    b = np.zeros_like(j)
    if "b" in streams or "bm" in streams:
        for idx, (v1, v2) in enumerate(COCO_BONE_PAIRS):
            b[:, :, idx, :2] = j[:, :, v1, :2] - j[:, :, v2, :2]
            b[:, :, idx, 2] = 0.5 * (j[:, :, v1, 2] + j[:, :, v2, 2])
        if "b" in streams:
            feats["b"] = b.copy()

    # Joint motion
    if "jm" in streams:
        jm = np.zeros_like(j)
        jm[:, :-1, :, :2] = np.diff(j[:, :, :, :2], axis=1)
        jm[:, :-1, :, 2] = 0.5 * (j[:, :-1, :, 2] + j[:, 1:, :, 2])
        jm[:, -1, :, :] = 0.0

        if conf_mask is not None:
            mask = np.asarray(conf_mask, dtype=bool)
            if mask.ndim == 2:
                mask = mask[None, ...]  # shape (1, T, 17)
            # Imputation-jump masking: step k requires both k and k+1 to be confident
            valid_motion = mask[:, :-1, :] & mask[:, 1:, :]
            for k in range(T - 1):
                invalid_k = ~valid_motion[0, k]
                jm[0, k, invalid_k, :2] = 0.0

        feats["jm"] = jm

    # Bone motion
    if "bm" in streams:
        bm = np.zeros_like(b)
        bm[:, :-1, :, :2] = np.diff(b[:, :, :, :2], axis=1)
        bm[:, :-1, :, 2] = 0.5 * (b[:, :-1, :, 2] + b[:, 1:, :, 2])
        bm[:, -1, :, :] = 0.0

        if conf_mask is not None:
            mask = np.asarray(conf_mask, dtype=bool)
            if mask.ndim == 2:
                mask = mask[None, ...]  # shape (1, T, 17)
            for idx, (v1, v2) in enumerate(COCO_BONE_PAIRS):
                # A bone motion is valid only if both endpoints were confident in both adjacent frames
                valid_bone_step = (
                    mask[:, :-1, v1] & mask[:, 1:, v1] &
                    mask[:, :-1, v2] & mask[:, 1:, v2]
                )
                for k in range(T - 1):
                    if not valid_bone_step[0, k]:
                        bm[0, k, idx, :2] = 0.0

        feats["bm"] = bm

    return feats


def uniform_sample_and_format_gcn(
    feat_dict: Dict[str, np.ndarray],
    clip_len: int = 100,
    num_person: int = 2,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Uniformly samples features across time dimension and formats for GCN input."""
    formatted: Dict[str, torch.Tensor] = {}

    for k, v in feat_dict.items():
        T = v.shape[1]
        inds = np.mod(np.arange(clip_len), T)
        sampled = v[:, inds, :, :]  # shape (1, clip_len, 17, 3)

        out_tensor = np.zeros((1, num_person, clip_len, 17, 3), dtype=np.float32)
        out_tensor[0, 0] = sampled[0]
        # Slot 1 is zero-padded for absent second person

        formatted[k] = torch.from_numpy(out_tensor).to(device)

    return formatted


def build_mmaction2_tensors(
    pts_30_17: np.ndarray,
    conf_mask_30_17: Optional[np.ndarray] = None,
    img_shape: Tuple[int, int] = (720, 1280),
    streams: Sequence[str] = ("j", "jm"),
    norm_mode: str = "prenormalize2d",
    clip_len: int = 100,
    device: str = "cpu",
    compact_scale_factor: float = 1.25
) -> Dict[str, torch.Tensor]:
    """End-to-end preprocessing pipeline from 30-frame track array to MMAction2 GCN input tensors."""
    pts_norm = prenormalize_2d(
        pts_30_17,
        img_shape=img_shape,
        norm_mode=norm_mode,
        compact_scale_factor=compact_scale_factor
    )
    feats = gen_ske_feat(pts_norm, conf_mask=conf_mask_30_17, streams=streams)
    return uniform_sample_and_format_gcn(feats, clip_len=clip_len, num_person=2, device=device)
