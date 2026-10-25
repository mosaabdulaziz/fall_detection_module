# Phase 5 Design (`design-2.md`): Pretrained Two-Stream ST-GCN (`TSSTG` / `tsstg-model.pth`) + Lightweight Duration & Immobility Pipeline on `yolo26x-pose.pt`

## 1. One-Sentence Restatement
The system tracks 17 COCO body keypoints per person at `stride=1` using `yolo26x-pose.pt` + `ByteTrack`, feeds each 30-frame 14-node `coco_cut` skeleton sequence into the pretrained Le2i Two-Stream ST-GCN (`tsstg-model.pth`) to classify 7 actions (`Standing`, `Walking`, `Sitting`, `Lying Down`, `Stand up`, `Sit down`, `Fall Down`), and applies a lightweight per-frame state accumulator for `fall` candidates, `lying` duration persistence (`T_lying`), and prolonged `immobility` (`T_immobile`).

---

## 2. Minimal End-to-End Skeleton (Stage by Stage with Evidence IDs)

### Stage 1 — Multi-Person Pose Extraction & Tracking at `stride=1` (`S0`, `S12`)
- **Component**: Run Ultralytics `yolo26x-pose.pt` + built-in `ByteTrack` (`bytetrack.yaml`) at **`stride=1`** (`S12`, `main.py`, `E105b`), outputting per-person track ID `tid`, bounding box `xyxy` `(4,)` (`[x1, y1, x2, y2]`), and 17 COCO keypoints `kp` of shape `(17, 3)` (`x_pixel, y_pixel, conf`) at timestamp `t` (`E01`).
- **Why `stride=1` is pinned**: `tsstg-model.pth` (`S12`) was trained on consecutive 30-frame windows (`stride=1`) where the second stream `mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]` measures single-frame displacement (`E105`, `E105b`). Running at `stride=1` preserves the exact temporal scale of `mot` and the `kernel_size=9` temporal convolutions (`E105`).
- **Evidence IDs**: `E01` (`S0`), `E105`, `E105b` (`S12`).

### Stage 2 — 17-to-14 `coco_cut` Keypoint Adapter, Finite Sanitization, and 30-Frame Buffer (`S12` + `[UNTESTED DEPARTURE 2]`)
- **Component**: For each tracked person `tid`, maintain a rolling `deque(maxlen=30)` of 13-keypoint arrays `(13, 3)` (`S12`, `Detection/Tracker.py`, `E105b`):
  1. **17-to-13 Keypoint Slicing (`E102`)**: Slice the 17 COCO keypoints by removing the 4 eye/ear indices (`1, 2, 3, 4`):
     `kp_13 = kp[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :].copy()` $\rightarrow$ shape `(13, 3)`.
  2. **Finite Coordinate Sanitization (`[UNTESTED DEPARTURE 2]`)**: Because `yolo26x-pose.pt` outputs `(0.0, 0.0, 0.0)` when a keypoint is unlocalized (which would pull `xy_min` in `scale_pose` to `(0, 0)`) and because `NaN` values fed into `nn.BatchNorm1d`/`Conv2d` corrupt all 7 output logits to `NaN`:
     - Any keypoint in `kp_13` with `conf < conf_kp` (`0.3` from `E04`) or `(x == 0 and y == 0)` is imputed in-place before appending to `deque(maxlen=30)`:
       - First, copy the last valid `(x, y)` for that joint from the previous frame in the track's `deque` (with `conf` kept as its actual low value).
       - If no prior frame in the `deque` has a valid observation for that joint (e.g., at track frame 0), fall back to the current bounding-box center `((x1 + x2) / 2.0, (y1 + y2) / 2.0)` so `kp_13[:, :2]` is **strictly finite (`np.isfinite(kp_13).all() == True`)** and remains inside the person's bounding box `[x1, y1, x2, y2]`.
  3. **Per-Frame Min-Max Scaling `scale_pose` with Zero-Range Epsilon (`E104` + `[UNTESTED DEPARTURE 2]`)**:
     - Once `len(deque) == 30` (`E105b`), stack into `pts` of shape `(30, 13, 3)`.
     - Normalize `(x, y)` to `[-1, 1]` per frame $i \in \{0, \dots, 29\}$ using `scale_pose` (`E104`), guarding the denominator with `np.maximum(xy_max[i] - xy_min[i], 1e-6)` so degenerate single-point frames cannot divide by zero:
       `xy[i] = ((xy[i] - xy_min[i]) / np.maximum(xy_max[i] - xy_min[i], 1e-6)) * 2.0 - 1.0`.
       *(Note: `normalize_points_with_size` prior to `scale_pose` is omitted or passed exact `(width=frame.shape[1], height=frame.shape[0])` because positive linear scaling cancels identically in `(xy - xy_min) / (xy_max - xy_min)`, `E104`).*
  4. **14th Node (Center-Shoulder / Neck) Concatenation (`E103`)**:
     - `pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2.0, 1)), axis=1)` $\rightarrow$ shape `(30, 14, 3)`.
  5. **Two-Stream Tensor Construction (`E105`)**:
     - `pts_tensor = torch.tensor(pts, dtype=torch.float32).permute(2, 0, 1)[None, :]` $\rightarrow$ shape `(1, 3, 30, 14)`.
     - `mot_tensor = pts_tensor[:, :2, 1:, :] - pts_tensor[:, :2, :-1, :]` $\rightarrow$ shape `(1, 2, 29, 14)`.
- **Evidence IDs**: `E04` (`S0`), `E102`, `E103`, `E104`, `E105`, `E105b` (`S12`).

### Stage 3 — Pretrained Two-Stream ST-GCN (`TSSTG`, `tsstg-model.pth`) Action Classifier (`S1`, `S12`)
- **Component**: Run `TwoStreamSpatialTemporalGraph` (`Actionsrecognition/Models.py`, `Actionsrecognition/Utils.py` with `layout='coco_cut'`, `strategy='spatial'`, 14 nodes, center node `13`, symmetric adjacency normalization $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, 10 `st_gcn` layers per stream, and `tsstg-model.pth` pretrained on Le2i) once per frame for each track with `len(deque) == 30` (`E105b`).
- **Output (`E101`, `E105`, `E105b`)**:
  - 7-element score vector `out = model((pts_tensor, mot_tensor))[0]` (where `model.forward` applies `torch.sigmoid(out)`, `E105`) over:
    `0: 'Standing'`, `1: 'Walking'`, `2: 'Sitting'`, `3: 'Lying Down'`, `4: 'Stand up'`, `5: 'Sit down'`, `6: 'Fall Down'` (`E101`).
  - Predicted class index `pred_idx = int(out.argmax())` and action label `action_name = class_names[pred_idx]` (`E105b`).
- **Evidence IDs**: `E101`, `E105`, `E105b`, `E110a`, `E111a`, `E111b` (`S1`, `S12`).

### Stage 4 — Lightweight Per-Frame State Accumulator for `fall`, `lying`, and `immobility` Candidates (`S0`, `S12`, `[UNTESTED DEPARTURE]`)
Instead of `S0`'s `0.5s` batch re-evaluation loop over a full 30-second raw array (which would cause a `30x` Pandas rolling bottleneck at `stride=1`, `PO4`), Stage 4 restricts kinematic rolling calculations (`smooth_s=0.4s`, `motion` `1.0s`, `window_s=2.0s` drop/vdown) to a bounded `<= 3.5s` sliding window (`~105` frames) and updates per-track state incrementally in $O(1)$ on every frame (matching `S12`'s per-frame `TSSTG` execution `E105b` while retaining `S0`'s `hold_s=1.0s` hysteresis and `realert_s=10.0s` / `cooldown_s=5.0s` deduplication `E04`):

1. **Candidate `fall` Event (`priority='high'`) via Streaming Pending-Fall Latch (`PO2`) & Hybrid `TSSTG` + Kinematic Veto (`PO3`)**:
   - **Latch Activation (`pending_fall_t0 = t`, `post_fall_stable_s = 0.0`)**:
     - **Path A (`TSSTG` Direct Fall Trigger, `len(deque) == 30`, `E101`, `E105b` + `[UNTESTED DEPARTURE 3]`)**: `pred_idx == 6` (`'Fall Down'`) while the bounding-box/posture satisfies `S0`'s horizontal/low-posture sanity gate (`aspect >= 0.5` from `E04` `lying.aspect_in` or `angle > 35` from `E04` `lying.angle_in`, preventing `scale_pose`'s aspect-ratio normalization `E104` from firing `'Fall Down'` on an upright standing person with `aspect < 0.5` and `angle <= 35`).
     - **Path B (Hybrid Kinematic Collapse Trigger Not Vetoed by `TSSTG`, `len(deque) == 30`, `PO3`, `[UNTESTED DEPARTURE 3]`)**: `S0`'s kinematic drop trigger (`drop > 0.10`, `vdown > 0.15`, `low_hip < 0.45`, `angle > 55 | aspect > 0.9`, `E04`) fires **AND** `TSSTG` does **not** veto it as a controlled upright/seated activity (`pred_idx in (3, 6)`, i.e. `'Lying Down'` or `'Fall Down'`, whereas controlled `'Standing'`=0, `'Walking'`=1, `'Sitting'`=2, `'Stand up'`=4, `'Sit down'`=5 actively **veto** look-alike sitting/bending false alarms).
     - **Path C (Cold-Start Fallback for `< 30` Frames, `PO2`/`O5`, `[UNTESTED DEPARTURE 5]`)**: When `len(deque) < 30` (e.g., person falls immediately upon entering camera view or after a `ByteTrack` ID switch mid-fall), `S0`'s kinematic drop trigger (`drop > 0.10`, `vdown > 0.15`, `low_hip < 0.45`, `angle > 55 | aspect > 0.9`, `E04`) activates the pending-fall latch directly.
   - **Streaming Post-Fall Stability Accumulation (`PO2`, `post_s = 3.0s`, `stable_s = 0.6s`, `E04`)**:
     - On each subsequent frame `t` while `pending_fall_t0 is not None` and `t - pending_fall_t0 <= post_s` (`3.0s`):
       - When `len(deque) == 30`: if `pred_idx in (3, 6)` (`'Lying Down'` or `'Fall Down'`) and `motion <= max_motion` (`1.5` from `E04`), increment `post_fall_stable_s += dt`. If `pred_idx in (0, 1, 2, 4, 5)` with upright posture (`aspect < 0.5` and `angle < 20`) or `motion > max_motion`, reset `post_fall_stable_s = 0.0`.
       - When `len(deque) < 30`: if `(hip_rel < 0.45) and (angle > 55 or aspect > 0.9)` and `motion <= 1.5` (`E04`), increment `post_fall_stable_s += dt`; else reset `post_fall_stable_s = 0.0`.
     - As soon as `post_fall_stable_s >= stable_s` (`0.6s`), emit the candidate `fall` event (`t_start = pending_fall_t0`, `t_end = t`, `priority = 'high'`, `status = 'CANDIDATE_PENDING_OPERATOR_REVIEW'`) and clear `pending_fall_t0 = None`. If `t - pending_fall_t0 > post_s` (`3.0s`) expires first, clear `pending_fall_t0 = None`.

2. **Candidate `lying` Event (`priority='normal'`, AC #2)**:
   - **Trigger (`E101`, `E04`)**: Track the continuous duration `lying_dur_s` where `(len(deque) == 30 and pred_idx == 3 and (aspect >= 0.5 or angle > 35))` (or during `< 30` warmup frames, `S0`'s `lying_cond` `angle > 35 & aspect > 0.5`, `E04`) and `motion <= max_motion` (`1.0` from `E04`), allowing brief gaps up to `hold_s = 1.0s` (`E04`). When `lying_dur_s >= T_lying` (`2.0s`, `E04`), emit a candidate `lying` event (`status = 'CANDIDATE_PENDING_OPERATOR_REVIEW'`).

3. **Candidate `immobility` Event (`priority='normal'`, AC #3 — `[UNTESTED DEPARTURE 4]`)**:
   - **Scale Normalization Without 10s `ref_h` Collapse (`O3`)**: To prevent `S0`'s 10-second rolling `ref_h` from shrinking during prolonged sitting or lying (`E02`), each track maintains a **monotonically non-decaying upright reference scale** `ref_scale = max(ref_scale_prev, np.hypot(w, h))` where `np.hypot(w, h)` is the bounding-box diagonal (`sqrt(w^2 + h^2)`). Because a lying person's width `w` equals their upright height `h`, the bounding-box diagonal `hypot(w, h)` stays invariant whether the person starts the track standing, sitting, or already lying on the floor!
   - **Trigger (`O9`)**: Compute instantaneous keypoint speed normalized by `ref_scale`: `motion_diag = nanmean(||xy[t] - xy[t-1]||_2) / (ref_scale * dt)` (smoothed over `1.0s` as in `E02`, with tracking gaps `dt > hold_s` masked to `NaN`). When a tracked person remains stationary (`motion_diag <= max_immobile_motion`, default `0.05` diagonal-lengths/s `[DECISION]`) continuously (with `hold_s = 1.0s` gap tolerance) for the posture-differentiated configured duration (`T_immobile_lying = 10.0s` when in lying posture `pred_idx == 3` or `angle > 35 & aspect > 0.5`, vs. `T_immobile_seated = 15.0s` for non-lying postures `[DECISION]`), emit a candidate `immobility` event (`status = 'CANDIDATE_PENDING_OPERATOR_REVIEW'`).

---

## 3. Component-by-Component Ledger Justification Table

| Component | Justification / Evidence IDs | Status |
|---|---|---|
| `yolo26x-pose.pt` + `ByteTrack` multi-person 17-keypoint extraction at `stride=1` | `E01` (`S0`), `E105`, `E105b` (`S12`) | `[SOURCE]` |
| Slicing 17 COCO keypoints `[0, 5..16]` $\rightarrow$ 13 keypoints + neck midpoint node `13` $\rightarrow$ 14-node `coco_cut` graph | `E102`, `E103` (`S12`, `PoseEstimateLoader.py`, `ActionsEstLoader.py`) | `[SOURCE]` |
| Per-frame `[-1, 1]` `scale_pose` + `(1, 3, 30, 14)` `pts` and `(1, 2, 29, 14)` `mot` streams | `E104`, `E105` (`S12`, `ActionsEstLoader.py`, `pose_utils.py`) | `[SOURCE]` |
| `TwoStreamSpatialTemporalGraph` (10 `st_gcn` layers, symmetric $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, `tsstg-model.pth` 7-class Le2i weights, `out[0].argmax()`) | `E101`, `E105`, `E105b`, `E110a`, `E111a`, `E111b` (`S1`, `S12`) | `[SOURCE]` |
| Baseline hysteresis/post-fall/lying parameters (`conf_kp=0.3`, `post_s=3.0`, `stable_s=0.6`, `max_motion=1.5/1.0`, `T_lying=2.0`, `hold_s=1.0`, `cooldown_s=5.0`, `angle_in=35`, `aspect_in=0.5`) | `E04` (`S0`, `fall_monitor.py` L28-34) | `[SOURCE]` |
| Feeding `yolo26x-pose.pt` + `ByteTrack` keypoints into `tsstg-model.pth` (trained on `AlphaPose` + `SORT`) | `E01`, `E102`, `E103` | `[UNTESTED DEPARTURE 1]` |
| Finite keypoint imputation (forward-fill along track `deque`, bbox-center fallback at frame 0, `1e-6` denominator guard in `scale_pose`) | `E04`, `E104`, `E105` | `[UNTESTED DEPARTURE 2]` |
| Post-`TSSTG` posture sanity gate (`aspect >= 0.5` or `angle > 35`, plus post-fall `stable_s=0.6s` with `motion <= 1.5` from `E04`) | `E04`, `E104`, `E108` | `[UNTESTED DEPARTURE 3]` |
| Prolonged unusual `immobility` detector (`detect_immobility`) using orientation-invariant bbox-diagonal `ref_scale = max(ref_scale, hypot(w, h))` over `T_immobile` | `E02`, `E03`, `E04` | `[UNTESTED DEPARTURE 4]` |
| Cold-start `< 30` frame kinematic fallback (`S0` `detect_falls` / `lying_cond` when `len(deque) < 30`) | `E04`, `E105b` | `[UNTESTED DEPARTURE 5]` |

---

## 4. Components Deliberately Excluded (The Delete Pass)

1. **`S12`'s Legacy Detector (`Tiny-YOLO`), Pose Estimator (`AlphaPose / SPPE FastPose`), and Tracker (`SORT / DeepSORT`)**:
   - **Why Excluded**: Replaced by `yolo26x-pose.pt` + `ByteTrack` (`S0`, `E01`).
2. **`PYSKL` (`S2`) Secondary Plug-In Backend in the Active Pipeline**:
   - **Why Excluded**: Deleted after Critic Objection `O11` (Necessity check). `PYSKL`'s NTU-60/120 checkpoints (`E109a`) only contain `'falling down'` (`A43`) and lack `'Lying Down'` (`class 3` in `TSSTG`, `E101`), so including `PYSKL` as an unintegrated secondary backend added dead complexity.
3. **`normalize_points_with_size` Prior to `scale_pose` (`O10`)**:
   - **Why Excluded**: Deleted after Critic Objection `O10`. Because `scale_pose` (`E104`) normalizes each frame's `x` and `y` independently using `((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1`, any prior linear division by `(width, height)` cancels out algebraically and introduces a `(height, width)` vs. `(width, height)` swap bug if misused.
4. **33-Joint 3D MediaPipe BlazePose Hypergraph-Mamba (`S4` FLASH) & Unreleased GCNs (`S3`, `S6`, `S7`)**:
   - **Why Excluded**: No runnable model code or pretrained `.pth` weights exist in their repositories (`S3.md`, `S4.md`, `S6.md`, `S7.md`).

---

## 5. Paper / Code Disagreements and How Each Is Resolved

1. **ST-GCN Layer Count (`9 layers` in `S1` Paper vs. `10 layers` in `S1` and `S12` Code — `E111a`)**:
   - **Resolution**: **Follow the code (10 layers)** so `tsstg-model.pth` loads without state_dict mismatch.
2. **Graph Adjacency Normalization (`normalize_digraph` in `S1` Code vs. `S12` Code — `E111b`)**:
   - **Resolution**: **Follow `S12` Code (`D^{-1/2} A D^{-1/2}`)** so graph convolutions match `tsstg-model.pth` training.

---

## 6. Faithfulness Declaration (`[UNTESTED DEPARTURE]` Flags & Accepted Limitations)

1. **`[UNTESTED DEPARTURE 1]` — Detector/Pose/Tracker Swap (`yolo26x-pose.pt` + `ByteTrack` $\rightarrow$ `tsstg-model.pth`)**:
   - `tsstg-model.pth` was trained on AlphaPose 13-keypoint sequences tracked by SORT on Le2i (`Coffee room`, `Home`) (`E101`, `E102`). Feeding `yolo26x-pose.pt` keypoints at `stride=1` is unvalidated in the literature and must be verified on Le2i test videos.
2. **`[UNTESTED DEPARTURE 2]` — Finite Keypoint Imputation Before `scale_pose`**:
   - Forward-filling low-confidence (`conf < 0.3`) or `(0, 0)` keypoints along the track's `deque` (with bbox-center fallback at frame 0 and `1e-6` denominator epsilon in `scale_pose`) prevents `NaN` propagation in `BatchNorm1d`/`Conv2d` and prevents `(0, 0)` corner distortion (`O2`).
3. **`[UNTESTED DEPARTURE 3]` — Post-`TSSTG` Posture & Motion Stability Gate (`E04`)**:
   - Requires `aspect >= 0.5` or `angle > 35` when `TSSTG` predicts `'Fall Down'`, followed within `post_s = 3.0s` by `stable_s = 0.6s` of `'Lying Down'` / `'Fall Down'` with `motion <= 1.5` (`E04`), guarding against `scale_pose`'s aspect-ratio loss (`E104`) and active floor work (`O6`).
4. **`[UNTESTED DEPARTURE 4]` — Orientation-Invariant `ref_scale` & Prolonged Immobility Detector (`detect_immobility`)**:
   - Uses `ref_scale = max(ref_scale_prev, hypot(w, h))` to eliminate the 10-second `ref_h` rolling collapse (`O3`, `E02`) and triggers candidate `immobility` when `motion_diag <= max_immobile_motion` persists for `T_immobile`.
5. **`[UNTESTED DEPARTURE 5]` — Cold-Start `< 30` Frame Kinematic Fallback**:
   - Runs `S0`'s `detect_falls` / `lying_cond` (`E04`) on tracks with `< 30` frames so falls occurring immediately upon entry or after a mid-fall `ByteTrack` ID switch (`O5`) are still caught.
6. **Surviving Accepted Limitations (from Critic Objections `O6` & `O9`)**:
   - **Limitation L1 (Look-Alike Generalization & Limb Occlusion, `O6`)**: `tsstg-model.pth` was trained only on Le2i (`Coffee room`, `Home`) across 7 classes (`Standing`, `Walking`, `Sitting`, `Lying Down`, `Stand up`, `Sit down`, `Fall Down`, `E101`) without explicit training classes for kneeling or deep bending, and coordinate-based GCNs lose accuracy (`-14.3` pp in `E108`) when limb keypoints are occluded. While `[UNTESTED DEPARTURE 3]` (post-fall `stable_s = 0.6s` lying stability + `motion <= 1.5`) suppresses active floor work and transient bends that return upright within `0.6s`, sustained motionless kneeling/crouching with occluded ankles can still trigger a candidate alert.
   - **Limitation L2 (Seated/Standing Stillness vs. Pathological Immobility, `O9`)**: Because Acceptance Criterion #3 defines immobility purely by low observable movement over a configured duration (`T_immobile`), any person sitting or standing completely still for `>= T_immobile` will trigger a candidate `immobility` event unless `T_immobile` is set sufficiently high for seated/standing postures (e.g., `T_immobile_seated = 60.0s–300.0s` vs. `T_immobile_lying = 10.0s`).

---

## 7. Resolved Decisions for `plan.md`

1. **Frame Stride**: Fixed to `stride=1` (`E105b`, resolving `[UNKNOWN 1]` and `O4`).
2. **Action Selection Rule**: Uses `pred_idx = int(out[0].argmax())` (`E105b`, resolving `[UNKNOWN 2]` and `O8`), optionally requiring `out[0, pred_idx] >= min_action_score` (default `0.5` `[DECISION]`).
