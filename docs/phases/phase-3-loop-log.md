# Phase 3 — Implementation Loop Log (Updated with Adversarial Remediations)

**Date**: 2026-10-22  
**Environment**: Windows 11, Python 3.14.4, PyTorch CUDA 12.6, Ultralytics 8.4.183, MMAction2 1.2.0, MMCV-Lite 2.1.0, ByteTrack (`lapx`).  
**Working Directory / Storage Constraint**: `F:\fall_detection\` (`STORAGE-1` fully respected; zero artifacts on `C:`).

---

## 1. Initial Implementation Loop (Tasks 1 through 5)
- **Task 1: ST-GCN Backbone & Loader** ([`tsstg_model.py`](file:///F:/fall_detection/tsstg_model.py)): Reused `mmaction.models.backbones.stgcn.STGCN`, COCO-17 directed graph partition `(3, 17, 17)`, SHA-256 weight registry, `fc_cls` remapping, and NTU60-to-7 class projection. (Tests T1–T4 passed).
- **Task 2: Skeleton Adapter** ([`keypoint_adapter.py`](file:///F:/fall_detection/keypoint_adapter.py)): 4-point imputation, velocity jump masking, 25 Hz resampling, `PreNormalize2D`, 100-frame tiling, and `M=2` zero-padding. (Tests T5–T8 passed).
- **Task 3: Transfer Gate** ([`transfer_gate.py`](file:///F:/fall_detection/transfer_gate.py)): Normalization mode evaluation, frozen-BatchNorm fine-tuning step. (Test T9 passed).
- **Task 4: Fall Monitor** ([`fall_monitor.py`](file:///F:/fall_detection/fall_monitor.py)): Track stitching, non-decaying `ref_scale`, anatomical knee angle, windowed displacement immobility, two-phase fall latch, conservative immobility thresholds (40s, 100s, 150s), and evidence packaging. (Tests T10–T15 passed).
- **Task 5: Benchmark Evaluation** ([`evaluate_benchmark.py`](file:///F:/fall_detection/evaluate_benchmark.py)): Le2i parser, benchmark metrics calculation, CSV/JSON manifest runner. (Test T16 passed).

---

## 2. Independent Adversarial Audit & Defect Discovery
An independent zero-trust code audit was launched to aggressively hunt for contradictions and critical logic bugs. The auditor identified 5 active defects:
1. **ST-GCN 30-Second Blackout**: `resample_window_25hz()` checked for $>1.0\text{s}$ tracking gaps across the entire 30s buffer rather than the active 1.16s window.
2. **Ghost Immobility on Occluded Persons**: `compute_windowed_displacement()` returned `0.0` when no joints were visible, satisfying `disp <= 0.035` and falsely declaring stillness.
3. **Post-Fall Immobility State Leak**: `last_fall_cand_t` was never invalidated on recovery/upright movement, causing normal seated stillness to trigger post-fall alarms minutes or hours later.
4. **Unbounded Memory Leak in `ref_scale`**: `diag_history` appended every frame without trimming, causing $O(N)$ growth.
5. **Spatial Mismatch in Ground-Truth Matching**: Temporal-only matching allowed false alarms on bystanders to masquerade as True Positives.

---

## 3. Faithful Test-Driven Regression & Verification

Per user instruction, faithful adversarial unit tests were implemented first in [`tests/test_adversarial_critiques.py`](file:///F:/fall_detection/tests/test_adversarial_critiques.py) without touching the code.
- **Initial Run Against Unmodified Code**: **5 FAILED, 1 PASSED**, proving that the auditor's findings were real, active defects.

### Applied Surgical Remediations:
1. In [`keypoint_adapter.py`](file:///F:/fall_detection/keypoint_adapter.py): isolated gap inspection strictly to the trailing window and intersecting boundary frames:
   ```python
   first_in_win = int(np.searchsorted(ts, t_win_start))
   slice_start = max(0, first_in_win - 1)
   ts_win = ts[slice_start:]
   if len(ts_win) < 2 or np.any(np.diff(ts_win) > max_gap_s):
       return None
   ```
2. In [`fall_monitor.py`](file:///F:/fall_detection/fall_monitor.py): returned `float(np.nan)` from `compute_windowed_displacement()` when `not excursions`, ensuring occluded frames evaluate `is_still = False`.
3. In [`fall_monitor.py`](file:///F:/fall_detection/fall_monitor.py): invalidated `d['last_fall_cand_t'] = None` when the person recovers an upright stance (`curr_hr > 0.40` or `curr_angle < 35.0`) or engages in active movement (`curr_motion > 1.2` or `curr_hr > 0.35`).
4. In [`fall_monitor.py`](file:///F:/fall_detection/fall_monitor.py): bounded `diag_history` in `update_ref_scale` to a 60-second rolling window (`max_history = 1500`).
5. In [`evaluate_benchmark.py`](file:///F:/fall_detection/evaluate_benchmark.py): added `compute_box_iou()` and `match_fall_event_to_gt()`, enforcing spatial overlap ($\text{IoU} \ge 0.3$) when ground-truth boxes are available.
6. In [`requirements.txt`](file:///F:/fall_detection/requirements.txt): pinned `mmengine>=0.10.0`, `mmcv-lite>=2.1.0`, `mmaction2>=1.2.0`, `importlib-metadata>=8.0`.

---

## 4. Full Suite Regression Results
- **Command**: `F:\fall_detection\.venv\Scripts\python.exe -m pytest tests/ -v`
- **Total Tests**: **33 collected** (26 initial + 7 adversarial critique tests)
- **Outcome**: **33 PASSED**, 0 failed, 2 benign warnings.
- **Execution Time**: 26.79s.

---

## 5. User Code Revision & Additional Hardening
- **Kinematic Veto Decoupling** (`fall_monitor.py`): Decoupled `kinematic_veto = bool(controlled_bend or controlled_knee)` so geometric crouch/bend vetoes are active from frame 0 even in cold-start windows before ST-GCN action probabilities become available (`has_30 = False`).
- **Path C Geometry Clarification**: Documented that Path C's fallen posture geometry (`angle > 55` or `aspect > 0.90` or `head < 0`) is mutually exclusive with controlled bend geometry (`angle < 45`, `aspect < 0.85`, `head > 0.30`).
- **Explicit Finite Displacement Guard**: Added `np.isfinite(disp_win)` to `is_still = np.isfinite(disp_win) and (disp_win <= disp_thresh)`.
- **Critique 7 Verification Test** (`tests/test_adversarial_critiques.py`): Added `test_critique_7_path_c_geometry_precludes_controlled_bend` verifying that true cold-start falls are detected while controlled sit-downs are rejected (**PASSED**).
