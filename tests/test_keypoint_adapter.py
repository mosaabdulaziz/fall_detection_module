"""T5-T8: keypoint_adapter.py - sanitising, 25 Hz resampling, PreNormalize2D -> GenSkeFeat -> UniformSample -> FormatGCNInput."""
import numpy as np
import pytest
import torch

import keypoint_adapter as KA  # ModuleNotFoundError == expected red
from synth import window_30, make_pose, STAND, IMG_SHAPE

PAIRS = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
         (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))


def _moving_j(T=30, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.5, 0.5, size=(1, 17, 3)).astype(np.float32)
    base[..., 2] = rng.uniform(0.4, 0.6, size=(1, 17)).astype(np.float32)
    drift = rng.uniform(-0.02, 0.02, size=(T, 17, 3)).astype(np.float32)
    drift[..., 2] = rng.uniform(-0.01, 0.01, size=(T, 17))        # time-varying score: catches score[k] vs mean(k,k+1)
    return base + np.cumsum(drift, axis=0)


# ---------------------------------------------------------------- T5
def test_prenormalize2d_and_gen_ske_feat_coco_17_modalities():
    assert KA.COCO_BONE_PAIRS == PAIRS
    h, w = IMG_SHAPE
    pts = np.zeros((2, 17, 3), dtype=np.float32)
    pts[0, :, :2] = (w / 2, h / 2); pts[1, :, :2] = (w, h); pts[:, :, 2] = 0.7
    out = KA.prenormalize_2d(pts, img_shape=IMG_SHAPE)
    np.testing.assert_allclose(out[0, :, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[1, :, :2], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[..., 2], 0.7, atol=1e-6)

    j = _moving_j()
    f = KA.gen_ske_feat(j, None, streams=("j", "jm", "b", "bm"))
    assert set(f) == {"j", "jm", "b", "bm"} and all(v.shape == (1, 30, 17, 3) for v in f.values())
    np.testing.assert_allclose(f["j"][0], j, atol=1e-6)
    b = np.zeros_like(j)
    for v1, v2 in PAIRS:
        b[:, v1, :2] = j[:, v1, :2] - j[:, v2, :2]
        b[:, v1, 2] = 0.5 * (j[:, v1, 2] + j[:, v2, 2])
    np.testing.assert_allclose(f["b"][0], b, atol=1e-6)
    for key, src in (("jm", j), ("bm", b)):
        m = np.zeros_like(src)
        m[:-1, :, :2] = np.diff(src[:, :, :2], axis=0)
        m[:-1, :, 2] = 0.5 * (src[:-1, :, 2] + src[1:, :, 2])
        np.testing.assert_allclose(f[key][0], m, atol=1e-6)
        assert np.all(f[key][0, -1] == 0.0)


# ---------------------------------------------------------------- T6
def test_finite_imputation_and_imputation_jump_masking_in_jm_and_bm():
    prev = make_pose(**STAND)
    kp = prev.copy() + 5.0
    kp[1, 2] = 0.1                       # low confidence
    kp[2, :2] = 0.0                      # detector "no point"
    kp[3, 0] = np.nan                    # non-finite
    box = np.array([100, 200, 300, 600], dtype=np.float32)
    clean, conf = KA.sanitize_keypoints_17(kp, box, prev, conf_kp=0.3)
    assert np.isfinite(clean).all()
    assert conf.tolist()[:5] == [True, False, False, False, True]
    np.testing.assert_allclose(clean[1:4, :2], prev[1:4, :2])
    np.testing.assert_allclose(clean[1:4, 2], 0.3)
    clean0, conf0 = KA.sanitize_keypoints_17(kp, box, None, conf_kp=0.3)
    np.testing.assert_allclose(clean0[1:4, :2], [[200.0, 400.0]] * 3)
    assert np.isfinite(clean0).all() and not conf0[1:4].any()

    # forward-filled joint 5 snaps to its real, distant position at frame 15
    j = _moving_j(); mask = np.ones((30, 17), dtype=bool); mask[:15, 5] = False
    j[:15, 5, :2] = j[0, 5, :2]; j[15:, 5, :2] += 0.8
    f = KA.gen_ske_feat(j, mask, streams=("j", "jm", "b", "bm"))
    assert np.all(f["jm"][0, :15, 5, :2] == 0.0)            # incl. the k=14 -> 15 snap
    assert np.any(f["jm"][0, 16:28, 5, :2] != 0.0)          # normal motion after re-detection survives
    assert np.all(f["bm"][0, :15, 5, :2] == 0.0)            # bone (5,0): endpoint 5
    assert np.all(f["bm"][0, :15, 7, :2] == 0.0)            # bone (7,5): endpoint 5
    assert np.any(f["bm"][0, :14, 16, :2] != 0.0)           # unrelated bone keeps its motion
    assert np.any(f["jm"][0, :14, 9, :2] != 0.0)


# ---------------------------------------------------------------- T7
def test_uniform_sample_100_order_and_format_gcn_m2_zero_pad():
    j = _moving_j()
    feats = KA.gen_ske_feat(j, np.ones((30, 17), dtype=bool), streams=("j", "jm", "b", "bm"))
    out = KA.uniform_sample_and_format_gcn(feats, clip_len=100, num_person=2)
    inds = np.mod(np.arange(100), 30)
    for k, x in out.items():
        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
        assert tuple(x.shape) == (1, 2, 100, 17, 3), k
        assert torch.all(x[0, 1] == 0)                       # absent 2nd person is zero-padded
        np.testing.assert_allclose(x[0, 0].numpy(), feats[k][0][inds], atol=1e-6)
    jm = out["jm"][0, 0, :, :, :2].abs().sum(dim=(1, 2)).numpy()
    zero_frames = np.flatnonzero(jm == 0)
    # motion computed BEFORE tiling: only the (zero-by-definition) last source frame is zero
    assert set(zero_frames.tolist()) == {i for i in range(100) if inds[i] == 29}


# ---------------------------------------------------------------- T8
def _traj(dt, T_end=2.0):
    ts = np.arange(0, T_end + 1e-9, dt)
    base = make_pose(**STAND)
    kps, masks = [], []
    for t in ts:
        k = base.copy(); k[:, 0] += 50.0 * t; k[:, 1] += 20.0 * t
        kps.append(k); masks.append(np.ones(17, dtype=bool))
    return list(ts), kps, masks


def test_fps_resampling_to_25hz_and_gap_reset_policy():
    a_pts, a_mask = KA.resample_window_25hz(*_traj(0.08))     # 12.5 fps
    b_pts, b_mask = KA.resample_window_25hz(*_traj(0.04))     # 25 fps
    assert a_pts.shape == (30, 17, 3) and a_mask.shape == (30, 17)
    np.testing.assert_allclose(a_pts, b_pts, atol=1e-4)
    assert a_mask.all() and b_mask.all()
    ta = KA.build_mmaction2_tensors(a_pts, a_mask, img_shape=IMG_SHAPE, streams=("j", "jm"))
    tb = KA.build_mmaction2_tensors(b_pts, b_mask, img_shape=IMG_SHAPE, streams=("j", "jm"))
    for k in ("j", "jm"):
        assert tuple(ta[k].shape) == (1, 2, 100, 17, 3)
        np.testing.assert_allclose(ta[k].numpy(), tb[k].numpy(), atol=1e-4)

    ts, kps, masks = _traj(0.04)
    assert KA.resample_window_25hz(ts[:5], kps[:5], masks[:5]) is None               # < 6 observations
    assert KA.resample_window_25hz(ts[:20], kps[:20], masks[:20]) is None            # span 0.76 s < 1.16 s
    gap_ts = ts[:31] + [2.4, 2.44]                                                    # 1.2 s hole > hold_s
    assert KA.resample_window_25hz(gap_ts, kps[:33], masks[:33]) is None
