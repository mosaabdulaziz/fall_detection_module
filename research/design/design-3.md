# Phase 5/6 Revised Design (`design-3.md`): OpenMMLab MMAction2 / PYSKL 17-COCO-Keypoint Multi-Stream ST-GCN (`MMAction2TSSTG`) + Transfer Gate, Lower-Limb/Action Veto, Windowed Displacement Immobility & Event-Evidence Pipeline on `yolo26x-pose.pt`

## 1. One-Sentence Restatement
The system extracts 17 native COCO body keypoints per person using `yolo26x-pose.pt` (`ultralytics >= 8.4.0`) + `ByteTrack` with spatial-temporal track stitching (`stitch_s = 1.5s`), resamples each track onto a 25.0 Hz (`dt = 0.04s`) 30-frame window with imputation-jump masking formatted through OpenMMLab MMAction2's exact `PreNormalize2D -> GenSkeFeat -> UniformSampleFrames(100) -> FormatGCNInput(M=2)` pipeline for the Apache-2.0 licensed **MMAction2 v1.0 / PYSKL 17-COCO-keypoint Multi-Stream ST-GCN** (`RecognizerGCN`, replacing `yysijie/st-gcn`), and combines smoothed action probabilities (`A43: falling` vs. `A8: sitting down`, `A6: pickup`, `A16/A17: shoe kneeling`, `A9: standing up`) with lower-limb articulation (`knee_angle`) and 2.0-second windowed confident-joint displacement to emit operator-reviewable candidate `fall`, `lying`, and `immobility` events with structured evidence.

---

## 2. Minimal End-to-End Skeleton (Stage by Stage with Evidence IDs)

### Stage 1 — Multi-Person Pose Extraction (`ultralytics >= 8.4.0`), `ByteTrack`, and Spatial-Temporal Track Stitching (`S0`, `FO2`, `FO6`)
- **Component**:
  - Run Ultralytics `yolo26x-pose.pt` (`ultralytics >= 8.4.0`, where YOLO26 pose weights were introduced) + built-in `ByteTrack` (`bytetrack.yaml`, `persist=True`, `conf_det=0.25`, `vid_stride=1`), outputting frame timestamp `t`, track IDs `ids`, bounding boxes `boxes` `(N, 4)` (`[x1, y1, x2, y2]`), and all 17 COCO keypoints `kps` `(N, 17, 3)` (`x_pixel, y_pixel, conf`) (`E01`, `E05`).
  - **Spatial-Temporal Track Stitching (`stitch_tracks`, `FO6`, `[UNTESTED DEPARTURE 5]`)**: When `ByteTrack` assigns a new track ID `tid` not currently active, check recently lost tracks (`0 < t - lost.last_t <= stitch_s` with `stitch_s = 1.5s`) whose last bounding-box center lies within `stitch_dist_scale * lost.ref_scale` (`0.8 * ref_scale`). If matched, transfer the lost track's state (`ref_scale`, `pending_fall_t0`, `post_fall_stable_s`, `t_hist`, `box_hist`, `kp_hist`) to `tid` so mid-fall ID switches do not lose the pre-switch drop history.
- **Evidence IDs**: `E01`, `E04`, `E05` (`S0`).

### Stage 2 — FPS-Aware 25 Hz Resampling, Native 17-Keypoint MMAction2 `PreNormalize2D` + `GenSkeFeat`, Gap Reset, and Imputation-Jump Masking (`S2`, `E204`, `FO7`)
- **Component**:
  1. **Track Gap Reset Policy (`FO7`)**: If a track experiences a detection gap `t - last_t > hold_s` (`1.0s` from `E04`), reset its temporal ST-GCN buffer so stale pre-gap poses are never differenced against post-gap poses.
  2. **FPS-Aware Time Resampling to `25.0 Hz` (`FO7`, `[UNTESTED DEPARTURE 2]`)**: Once a track spans `>= 1.16s` with `>= 6` valid observations and no gap `> hold_s`, resample the last `1.16s` onto 30 uniform timestamps `t_grid = np.linspace(t - 1.16, t, 30)` (`25.0 Hz`, `dt = 0.04s`) across all **17 native COCO keypoints (`V = 17`, indices `0..16`)** — eliminating the old `GajuuzZ` 17-to-13 slicing and synthetic 14th neck node (`E202`, `E204`).
  3. **Confidence Masking & Causal Finite Imputation (`O2`, `FO7`)**:
     - Record boolean mask `is_confident = np.isfinite(kp_17[..., 0]) & np.isfinite(kp_17[..., 1]) & (kp_17[..., 2] >= conf_kp) & ~((kp_17[..., 0] == 0.0) & (kp_17[..., 1] == 0.0))` (`conf_kp = 0.3` from `E04`, shape `(T, 17)`).
     - Impute non-confident joints via causal forward-fill along the track (falling back to current bbox center `((x1+x2)/2, (y1+y2)/2)` at track frame 0) so coordinates are strictly finite (`np.isfinite(kp_17).all()`).
  4. **MMAction2 `PreNormalize2D` (`mmaction/datasets/transforms/pose_transforms.py#L710-L745`, `E204`)**:
     - Using `img_shape = (h, w)` (or subject-centered padded crop `norm_mode="track_compact"` for wide-angle far-field CCTV), normalize `(x, y)` into `[-1, 1]`:
       $$x_{\text{norm}} = \frac{x - w/2}{w/2}, \quad y_{\text{norm}} = \frac{y - h/2}{h/2}$$
     - Unlike `GajuuzZ`'s per-frame `scale_pose` (which stretched every frame independently to `[-1, 1]^2` and erased global downward translation and aspect ratio), `PreNormalize2D` preserves true frame-relative vertical drop and body aspect ratio (`E109b`, `E204`).
  5. **MMAction2 `GenSkeFeat(dataset='coco', feats=['j', 'jm', 'b', 'bm'])` Executed BEFORE Frame Sampling (`E204`, `FO7`)**:
     - **Joint stream (`j`)**: shape `(1, T, 17, 3)` with channels `(x_norm, y_norm, score)`.
     - **Joint-Motion stream (`jm`, `ToMotion`, `pose_transforms.py#L864-L875`)**: shape `(1, T, 17, 3)` where `jm[:, :T-1, :, :2] = np.diff(j[:, :, :, :2], axis=1)`, `jm[:, :T-1, :, 2] = 0.5 * (j[:, :T-1, :, 2] + j[:, 1:, :, 2])`, and `jm[:, T-1] = 0.0`.
       - **Imputation-Jump Masking (`FO7`)**: For any frame transition `k -> k+1` and joint `v` where `not (is_confident[k, v] and is_confident[k+1, v])`, set `jm[:, k, v, :2] = 0.0` so a forward-filled joint snapping back to its real position on re-detection never injects a false velocity spike into `jm`.
     - **Bone stream (`b`, `JointToBone`, `pose_transforms.py#L790-L815`)**: shape `(1, T, 17, 3)` using the 17 COCO directed pairs `COCO_BONE_PAIRS = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))`, where `b[..., v1, :2] = j[..., v1, :2] - j[..., v2, :2]` and `b[..., v1, 2] = 0.5 * (j[..., v1, 2] + j[..., v2, 2])`.
     - **Bone-Motion stream (`bm`)**: `ToMotion` applied to `b`, with imputation-jump masking when either endpoint `(v1, v2)` is non-confident at `k` or `k+1`.
  6. **MMAction2 `UniformSampleFrames(clip_len=100, test_mode=True)` + `FormatGCNInput(num_person=2, mode='zero')` (`E204`)**:
     - Sample each generated stream from length `T` (`30`) to `clip_len = 100` using MMAction2's exact test-mode modulo loop indexing `inds = np.mod(np.arange(100), T)` when `T < 100` (`pose_transforms.py#L1083-L1123`). *(Executing `GenSkeFeat` before `UniformSampleFrames` ensures `jm` and `bm` never take differences across duplicated frame indices.)*
     - Zero-pad single-person tensor `(1, 100, 17, 3)` along the person axis to `num_person = 2` (`(1, 2, 100, 17, 3)`) via `FormatGCNInput(num_person=2, mode='zero')` (`formatting.py#L425-L445`) so `BatchNorm` and `GCNHead` (`x.mean(dim=1)`) match the exact NTU-60 training/validation distribution.
- **Evidence IDs**: `E04` (`S0`), `E109b`, `E202`, `E204` (`S2`).

### Stage 3 — Standalone OpenMMLab MMAction2 / PYSKL `RecognizerGCN` (`tsstg_model.py`) & Transfer Gate (`S2`, `E201`–`E205`, `FO1`–`FO3`, `FO8`)
- **Component**:
  - Replace `yysijie/st-gcn` / `GajuuzZ` with a zero-`mmcv`/zero-`mmengine` pure-PyTorch port of OpenMMLab MMAction2's `Graph`, `unit_gcn`, `unit_tcn`, `STGCNBlock`, `STGCN`, `GCNHead`, and `RecognizerGCN` (`mmaction/models/utils/graph.py`, `mmaction/models/utils/gcn_utils.py`, `mmaction/models/backbones/stgcn.py`, `mmaction/models/heads/gcn_head.py`, `E202`, `E203`):
    - `Graph(layout='coco', mode='stgcn_spatial')`: `num_node = 17`, `center = 0`, 16 `inward` edges `[(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]`, asymmetric `normalize_digraph(adj)` ($\mathbf{A}\mathbf{D}^{-1}$), and `(3, 17, 17)` spatial partition `[a_close(hop=0), a_close(hop=1), a_further(hop=1)]` where `a_close` is assigned when `hop_dis[j, center] >= hop_dis[i, center]` (`E202`).
    - Pre-load adjacency buffer verification (`T2`, `FO1`): assert `torch.allclose(model.backbone.gcn[0].gcn.A, state_dict['backbone.gcn.0.gcn.A'], atol=1e-6)` **before** calling `load_state_dict(..., strict=True)`.
    - Seamless support for both **MMAction2 v1.0** checkpoints (`cls_head.fc.{weight,bias}`) and **PYSKL** checkpoints (`cls_head.fc_cls.{weight,bias}` remapped to `cls_head.fc.{weight,bias}`), verified via pinned SHA-256 checksums on `F:\fall_detection\weights` and loaded with `torch.load(..., weights_only=True)` (`E201`, `FO2`).
  - **Multi-Stream Fusion & Seamless 60-to-7 Class Projection (`MMAction2TSSTG` / `TSSTGClassifier`, `E201`, `E205`)**:
    - Supports configurable stream ensembles: Two-Stream `("j", "jm")` (spatial + motion) or `("j", "b")` (`92.12%` Top-1), Three-Stream `("j", "jm", "b")`, or Four-Stream `("j", "b", "jm", "bm")` (`92.34%` Top-1, weights `2:2:1:1`).
    - Averages per-stream `F.softmax(logits, dim=-1)` (`(B, 60)`) and outputs both:
      1. `ntu60_probs: np.ndarray` `(60,)` over `label_map_ntu60.txt` (`E205`), and
      2. `probs_7: np.ndarray` `(7,)` over `ACTION_CLASSES = ('Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down')` for drop-in compatibility with Stage 4:
         - `probs_7[6]` (`Fall Down`): `ntu60_probs[42]` (`A43: "falling"`)
         - `probs_7[5]` (`Sit down` / Controlled Bend & Kneel): `ntu60_probs[[5, 7, 15, 16]].sum()` (`A6: "pickup"`, `A8: "sitting down"`, `A16: "wear a shoe"`, `A17: "take off a shoe"`)
         - `probs_7[4]` (`Stand up`): `ntu60_probs[8]` (`A9: "standing up (from sitting position)"`)
         - `probs_7[1]` (`Walking`): `ntu60_probs[[25, 26, 41, 58, 59]].sum()` (`"hopping"`, `"jump up"`, `"staggering"`, `"walking towards/apart"`)
         - `probs_7[2]` (`Sitting`): `ntu60_probs[[10, 11, 28, 29]].sum()` (`"reading"`, `"writing"`, `"playing with phone/tablet"`, `"typing on a keyboard"`)
         - `probs_7[3]` (`Lying Down`): `0.0` in raw NTU-60 head (or fine-tuned head probability), complemented by `lying_cond` in Stage 4
         - `probs_7[0]` (`Standing`): remaining probability mass `max(0.0, 1.0 - probs_7[1:].sum())`.
  - **Transfer Verification & Fine-Tuning Gate (`transfer_gate.py`, `FO3`)**:
    - Evaluates `yolo26x-pose.pt` skeletons on held-out fall, sitting, bending (`pickup`), and kneeling (`shoe`) clips across checkpoint families (`mmaction2` vs. `pyskl`), stream combinations (`j+jm`, `j+b`, `j+b+jm+bm`), and normalization modes (`prenormalize2d` vs. `track_compact`), and provides `finetune_tsstg_on_yolo26()` if held-out macro-F1 `< 0.80`.
- **Evidence IDs**: `E201`, `E202`, `E203`, `E204`, `E205` (`S2`).

### Stage 4 — Hybrid Fall, Lying & Windowed-Displacement Immobility State Machine with Event Evidence (`S0`, `S2`, `FO4`, `FO5`, `FO9`, `FO11`)
Preserves `S0`'s batch streaming interface `FallMonitor.update(t, ids, boxes, kps)` (`E05`) and convention `vdown = -np.diff(hip_rel) / dt` (**downward = positive**, `vpk > p['vdown'] = 0.15`, `E05`):

1. **Temporal Smoothing & Lower-Limb Articulation (`FO4`, `E01`, `E205`)**:
   - Smooth the 7-class projected probability vector `probs_7` over `smooth_s = 0.4s` (`probs_smooth`) and require `probs_smooth[pred_idx] >= min_action_score` (`0.50`).
   - Extract lower-limb keypoints omitted by `S0` (`E01`): knees `[13, 14]` and ankles `[15, 16]`. Compute interior `knee_angle` (hip–knee–ankle angle in degrees) and `hip_to_ankle_rel = (ankles_y - hips_y) / ref_scale`.
   - Define **Controlled-Action & Lower-Limb Veto (`controlled_veto`)**: `True` when `(has_30 and probs_smooth[[0, 1, 2, 4, 5]].max() > max(probs_smooth[3], probs_smooth[6]))` (which directly includes NTU-60 `sitting down` `A8`, `pickup` `A6`, `wear/take off a shoe` `A16/A17`, and `standing up` `A9` in `probs_smooth[5]` and `probs_smooth[4]`) **OR** when the person is in a controlled bend/kneel/crouch (`angle < 45` and `aspect < 0.85` and `head_rel > 0.30` and `vpk <= 0.30`).

2. **Candidate `fall` Event (`priority='high'`) via Two-Phase Pending-Fall Latch (`PO2`, `FO4`)**:
   - **Latch Activation (`pending_fall_t0 = t`, `post_fall_stable_s = 0.0`)**:
     - **Path A (`MMAction2TSSTG` Fall Trigger, `has_30`)**: `pred_idx == 6` (`A43: falling`) and `probs_smooth[6] >= min_action_score` (`0.50`) and `not controlled_veto` and a measurable downward descent (`drop > 0.08` or `vpk > 0.12` over `window_s = 2.0s`) and fallen geometry (`angle > 45` or `aspect > 0.85` or `hip_rel < 0.45`).
     - **Path B (Hybrid Kinematic Collapse Trigger, `has_30`)**: `drop > 0.10` and `vpk > 0.15` (`E04`, `E05`) and `not controlled_veto` and `(probs_smooth[3] + probs_smooth[6] >= min_action_score)` and `(hip_rel < 0.45) and (angle > 55 or aspect > 0.9 or head_rel < 0.0)`.
     - **Path C (`< 30` Frame Cold-Start Fallback, `not has_30`)**: `drop > 0.10` and `vpk > 0.15` and `(hip_rel < 0.45) and (angle > 55 or aspect > 0.9 or head_rel < 0.0)` (`E04`).
   - **Strict Post-Fall Lying Stability Check (`post_s = 3.0s`, `stable_s = 0.6s`, `FO4`)**:
     - On each frame in `(pending_fall_t0, pending_fall_t0 + 3.0s]`, stability requires a **low/horizontal lying state**:
       `is_lying_posture = (hip_rel < 0.45) and (angle > 55 or aspect > 0.9 or head_rel < 0.0)` **AND** `(not has_30 or not controlled_veto)` **AND** `motion <= max_motion` (`1.5`).
       *(Seated or leaning postures with `aspect <= 0.85` and `angle <= 45` can never satisfy post-fall stability even if `pred_idx == 6`!)*
     - Accumulate `post_fall_stable_s` across valid frames using `S0`'s `hold_runs` gap tolerance (`hold_s = 1.0s`). Once `post_fall_stable_s >= stable_s` (`0.6s`), emit candidate `fall` event and record `last_fall_t = t` on the track.

3. **Candidate `lying` Event (`priority='normal'`, AC #2, `FO11`)**:
   - Uses `S0`'s exact `hold_runs(t, cond, hold_s=1.0)` + `merge_runs(runs, cooldown_s=5.0)` semantics (`E04`, `E05`) where `cond = 1.0` when `(lying_cond == 1.0) and (not has_30 or not controlled_veto)`, `0.0` when `lying_cond == 0.0 or (has_30 and pred_idx in (0, 1, 4))`, and `NaN` in between. When a merged run satisfies `e - s >= T_lying` (`2.0s`) and `_mean_in(motion, t, s, e) <= max_motion` (`1.0`), emit candidate `lying` event.

4. **Candidate `immobility` Event (`priority='normal'`, AC #3 — Windowed Excursion on Confident Joints, `FO5`)**:
   - **Outlier-Clamped Non-Decaying Reference Scale (`ref_scale`, `FO5`)**: Maintain a rolling 60-second buffer of single-person bounding-box diagonals `d = hypot(w, h)`. Let `ref_scale` be the 90th percentile of `d` clamped to `<= 1.15 * median(d)` and floored by `ref_scale_init` (preventing the 10-second `ref_h` collapse `E02`).
   - **Windowed Net Excursion (`disp_win`, `FO5`)**: Over `immobile_win_s = 2.0s`, select only joints $j \in \{0,\dots,16\}$ with `>= 3` **confident (`conf >= conf_kp`, non-imputed)** observations in `[t - 2.0s, t]`. Compute each confident joint's maximum distance from its 2.0s window centroid $\bar{\mathbf{x}}_j$, take the median across confident joints, and divide by `ref_scale`:
     $$\text{disp\_win}(t) = \frac{\text{median}_{j \in \text{valid}} \max_{\tau \in [t-2.0, t]} \|\mathbf{x}_j(\tau) - \bar{\mathbf{x}}_j\|_2}{\text{ref\_scale}(t)}$$
   - **Differentiated Durations (`FO5`)**: `T_immobile_post_fall = 40.0s`, `T_immobile_lying = 100.0s`, `T_immobile_seated = 150.0s`. (**Conservative thresholds authorized during Phase 3 implementation loop** — original design values 5s/10s/120s were superseded to reduce nuisance alerts on slow-recovery and resting subjects.)

5. **Candidate Lifecycle, Deduplication & Structured Event Evidence (`FO9`, `FO11`)**:
   - **Deduplication (`E05`)**: `cooldown_s = 5.0s` suppresses overlapping buffer evaluations of the same incident (`ev['t_candidate'] <= last_cand[key] + cooldown_s`), while `realert_s = 60.0s` throttles repeat alerts of the same `(type, track)` (`t - emitted[key] < realert_s`).
   - **Operator Lifecycle (AC #4)**: Every emitted event includes `event_id: str`, `status: "candidate"`, and can be transitioned via `FallMonitor.review_event(event_id, status="acknowledged" | "confirmed" | "dismissed", operator_note="")`.
   - **Structured Event Evidence**: Every event carries an `evidence` dict with `clip_window: [t_start - 2.0, t_alert + 2.0]`, `snapshot_path`, `tsstg_probs` (7-class dict + top-5 `ntu60` classes), `kinematics` (`drop`, `vdown`, `hip_rel`, `angle`, `aspect`, `knee_angle`, `disp_win`, `motion`), and `keypoint_summary`.

---

## 3. Component-by-Component Ledger Justification Table

| Component | Justification / Evidence IDs | Status |
|---|---|---|
| `yolo26x-pose.pt` (`ultralytics >= 8.4.0`) + `ByteTrack` batch `update(t, ids, boxes, kps)` | `E01`, `E05` (`S0`), `FO2`, `FO9` | `[SOURCE]` |
| OpenMMLab MMAction2 / PYSKL `Graph(layout='coco', mode='stgcn_spatial')` with native 17 COCO keypoints (`V=17`), asymmetric `normalize_digraph` ($\mathbf{A}\mathbf{D}^{-1}$), and `(3, 17, 17)` spatial partition verified against `state_dict['backbone.gcn.0.gcn.A']` | `E201`, `E202`, `E203` (`S2`, `mmaction/models/utils/graph.py`), `FO1` | `[SOURCE]` |
| OpenMMLab MMAction2 preprocessing order: `PreNormalize2D` $\rightarrow$ `GenSkeFeat(dataset='coco', feats=['j','jm','b','bm'])` $\rightarrow$ `UniformSampleFrames(clip_len=100, test_mode=True)` $\rightarrow$ `FormatGCNInput(num_person=2, mode='zero')` | `E109b`, `E204` (`S2`, `mmaction/datasets/transforms/pose_transforms.py` & `formatting.py`) | `[SOURCE]` |
| NTU-60 action class mapping (`42: falling` vs. controlled look-alikes `7: sitting down`, `5: pickup`, `15/16: wear/take off a shoe`, `8: standing up`) | `E205` (`S2`, `tools/data/skeleton/label_map_ntu60.txt`) | `[SOURCE]` |
| Baseline kinematic definitions (`vdown` downward positive, `hold_runs`, `merge_runs`, `cooldown_s=5.0`, `realert_s=60.0`) | `E04`, `E05` (`S0`, `fall_monitor.py` L28–170), `FO11` | `[SOURCE]` |
| Transfer Verification & Fine-Tuning Gate (`transfer_gate.py`) testing `yolo26x-pose.pt` $\rightarrow$ `MMAction2TSSTG` on held-out clips | `E01`, `E201`–`E205`, `FO3` | `[UNTESTED DEPARTURE 1]` (Tested at Gate 1) |
| FPS-aware 25 Hz resampling, track gap `> hold_s` buffer reset, finite imputation, and imputation-jump masking in `jm`/`bm` before `UniformSampleFrames(100)` | `E04`, `E204`, `O2`, `FO7` | `[UNTESTED DEPARTURE 2]` |
| Smoothed `MMAction2TSSTG` 7-class projection + lower-limb `knee_angle` controlled-action veto & strict lying post-fall stability | `E01`, `E04`, `E05`, `E205`, `O6`, `PO2`, `PO3`, `FO4` | `[UNTESTED DEPARTURE 3]` |
| Windowed 2.0s confident-joint excursion (`disp_win`) + outlier-clamped 90th-percentile `ref_scale` for `detect_immobility` | `E02`, `E03`, `E04`, `O3`, `O9`, `FO5` | `[UNTESTED DEPARTURE 4]` |
| Spatial-temporal track stitching (`stitch_s = 1.5s`) + `< 30` frame cold-start kinematic fallback | `E04`, `E05`, `O5`, `FO6` | `[UNTESTED DEPARTURE 5]` |

---

## 4. Paper / Code Disagreements and How Each Is Resolved

1. **ST-GCN Layer Count (`9 layers` in `S1` Paper vs. `10 STGCNBlock stages` in `S1` and `S2` MMAction2/PYSKL Code — `E111a`, `E203`)**:
   - **Resolution**: **Follow the code (`num_stages = 10` in `mmaction/models/backbones/stgcn.py`)** so all 8 MMAction2 v1.0 and PYSKL checkpoints load with `strict=True` (`T1`).
2. **Graph Adjacency Normalization ($\Lambda^{-1/2}\mathbf{A}\Lambda^{-1/2}$ in `S1` Paper Eq. 3 vs. Asymmetric `normalize_digraph` $\mathbf{A}\mathbf{D}^{-1}$ in `S1` and `S2` MMAction2/PYSKL Code — `E111b`, `E202`, `FO1`)**:
   - **Resolution**: **Follow `mmaction/models/utils/graph.py` (`normalize_digraph` $\mathbf{A}\mathbf{D}^{-1}$ and `stgcn_spatial` `(3, 17, 17)` partition)** and verify `torch.allclose(model.backbone.gcn[0].gcn.A, sd['backbone.gcn.0.gcn.A'], atol=1e-6)` **before** calling `load_state_dict` (`T2`).
3. **Checkpoint Head Layer Key Name (`cls_head.fc` in MMAction2 v1.0 vs. `cls_head.fc_cls` in PYSKL — `E203`)**:
   - **Resolution**: Standardize on `cls_head.fc` in `GCNHead` and automatically remap `cls_head.fc_cls.{weight,bias} -> cls_head.fc.{weight,bias}` on checkpoint load so both MMAction2 v1.0 and PYSKL checkpoints load with `strict=True`.

---

## 5. Licensing, Provenance & Operational Constraints (`FO2`, `E201`)
- **OpenMMLab MMAction2 v1.0 & PYSKL Checkpoints (`S2`, `E201`)**: Licensed under **Apache-2.0** (permissive commercial & research license), hosted on official `https://download.openmmlab.com/...` (`HTTP 200 OK`), stored on `F:\fall_detection\weights`, verified via pinned SHA-256 digests, and loaded exclusively with `torch.load(..., weights_only=True)`.
- **Ultralytics (`ultralytics >= 8.4.0`)**: Licensed under **AGPL-3.0** (requires open-sourcing derivative works or acquiring an Ultralytics Enterprise License for commercial deployment).
- **Evaluation Datasets**: Le2i is distributed under **CC-BY-NC-SA-3.0** and NTU-RGB+D 60 is distributed under academic research terms (commercial deployment requires verifying dataset/fine-tuning licensing or evaluating on in-house CCTV recordings).
