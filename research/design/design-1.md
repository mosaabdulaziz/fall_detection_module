# Phase 5 Design (`design-1.md`): Pretrained Two-Stream ST-GCN (`TSSTG`) + Lightweight Temporal Persistence & Immobility Pipeline on `yolo26x-pose.pt`

## 1. One-Sentence Restatement
The system tracks 17 COCO body keypoints per person using `yolo26x-pose.pt` + `ByteTrack`, maps each 30-frame track window into a 14-node `coco_cut` graph classified into 7 actions (`Standing`, `Walking`, `Sitting`, `Lying Down`, `Stand up`, `Sit down`, `Fall Down`) by the pretrained Le2i Two-Stream ST-GCN (`tsstg-model.pth`), and applies a lightweight temporal duration and keypoint-displacement state machine to emit candidate `fall`, `lying`, and `immobility` alerts for operator review.

---

## 2. Minimal End-to-End Skeleton (Stage by Stage with Evidence IDs)

1. **Stage 1 — Multi-Person Pose Extraction & Tracking (`S0`)**:
   - **Component**: Ultralytics `yolo26x-pose.pt` + built-in `ByteTrack` (`bytetrack.yaml`) outputting per-person track ID `tid`, bounding box `xyxy` `(4,)`, and 17 COCO keypoints `(17, 3)` (`x, y, conf`) at each processed frame timestamp `t`.
   - **Evidence IDs**: `E01`, `E02`, `E03` (`S0`, [`fall_monitor.py`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py#L199-L211)).

2. **Stage 2 — 17-to-14 `coco_cut` Keypoint Adapter & 30-Frame Sliding Buffer (`S12`)**:
   - **Component**: For each tracked person `tid`, maintain a rolling `deque(maxlen=30)` of frames (`S12`, `Detection/Tracker.py`). On each frame:
     1. Slice the 17 COCO keypoints by removing the 4 eye/ear indices (`1, 2, 3, 4`) via `kp_13 = kp[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]` $\rightarrow$ shape `(13, 3)` (`E102`).
     2. Normalize `(x, y)` pixel coordinates by frame `(width, height)` (`normalize_points_with_size`) and scale each frame's `xy` to `[-1, 1]` using per-frame min/max (`scale_pose`: `((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1`) (`E104`).
     3. Append the 14th node (Index `13`, Center-Shoulder / Neck) by averaging Index `1` (Left Shoulder, COCO `5`) and Index `2` (Right Shoulder, COCO `6`): `pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)` $\rightarrow$ shape `(30, 14, 3)` (`E103`).
     4. Construct the position stream tensor `pts` of shape `(1, 3, 30, 14)` and motion stream tensor `mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]` of shape `(1, 2, 29, 14)` (`E105`).
   - **Evidence IDs**: `E102`, `E103`, `E104`, `E105` (`S12`, `PoseEstimateLoader.py`, `ActionsEstLoader.py`, `pose_utils.py`).

3. **Stage 3 — Pretrained Two-Stream ST-GCN (`TSSTG`, `tsstg-model.pth`) Action Classifier (`S1`, `S12`)**:
   - **Component**: Load `TwoStreamSpatialTemporalGraph` (`Actionsrecognition/Models.py`, `Actionsrecognition/Utils.py` with `layout='coco_cut'`, `strategy='spatial'`, 14 nodes, center node `13`, symmetric adjacency normalization $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, 10 `st_gcn` blocks per stream, and `tsstg-model.pth` pretrained on Le2i) to output a 7-class probability vector `probs` over:
     - `0: Standing`, `1: Walking`, `2: Sitting`, `3: Lying Down`, `4: Stand up`, `5: Sit down`, `6: Fall Down` (`E101`).
   - **Evidence IDs**: `E101`, `E103`, `E105`, `E110a`, `E111a`, `E111b` (`S1`, `S12`).

4. **Stage 4 — Lightweight Temporal Persistence, Immobility & Candidate Alert State Machine (`S0` + `[UNTESTED DEPARTURE]`)**:
   - **Component**: Maps the 7-class `TSSTG` probabilities + raw keypoint displacement over time `t` into the three required candidate event types (`fall`, `lying`, `immobility`) with cooldown/realert deduplication (`S0`, `FallMonitor`):
     - **Candidate `fall` Event (`priority='high'`)**: Triggered when `TSSTG` predicts `'Fall Down'` (`probs[6] >= p_fall_thresh`, or `'Fall Down'` is argmax above threshold) across `N_fall_frames` within a short window, `[UNTESTED DEPARTURE]` guarded by a minimum bounding-box/keypoint drop sanity check (`aspect >= min_fall_aspect` or post-fall `'Lying Down'` transition) so `scale_pose`'s aspect-ratio blindness (`E104`) cannot fire a false `'Fall Down'` on an upright standing person with noisy keypoints (`E108`).
     - **Candidate `lying` Event (`priority='normal'`)**: Triggered when `TSSTG` predicts `'Lying Down'` (`probs[3] >= p_lying_thresh` or argmax == `3`) continuously (allowing `hold_s` hysteresis gaps from `S0` `hold_runs`) for configured duration `T_lying` (AC #2).
     - **Candidate `immobility` Event (`priority='normal'`)**: `[UNTESTED DEPARTURE]` Triggered when a tracked person (in *any* non-walking posture: `'Standing'`, `'Sitting'`, or `'Lying Down'`) exhibits mean body-height-normalized keypoint displacement speed `motion <= max_immobile_motion` continuously for configured duration `T_immobile` (e.g., `10.0s`–`30.0s`), with tracking gaps (`dt > hold_s`) masked to `NaN` as in `S0` (`E02`, `E03`).
     - **Look-Alike Suppression (AC #5: Bending, Kneeling, Sitting, Floor Work)**: Handled natively by `TSSTG`'s explicit `'Sitting'`, `'Sit down'`, `'Stand up'`, `'Standing'`, and `'Walking'` classes (`E101`) combined with the ongoing keypoint motion check (`motion > max_motion`) during active floor work (`S0`).

---

## 3. Component-by-Component Ledger Justification Table

| Component | Justification / Evidence IDs | Status |
|---|---|---|
| `yolo26x-pose.pt` + `ByteTrack` multi-person 17-keypoint extraction | `E01` (`S0`, `fall_monitor.py` L199-211) | `[SOURCE]` |
| Slicing 17 COCO keypoints `[0, 5..16]` $\rightarrow$ 13 keypoints + neck midpoint node `13` $\rightarrow$ 14-node `coco_cut` graph | `E102`, `E103` (`S12`, `PoseEstimateLoader.py`, `ActionsEstLoader.py`) | `[SOURCE]` |
| Per-frame `[0, 1]` normalization + `[-1, 1]` `scale_pose` + `(1, 3, 30, 14)` `pts` and `(1, 2, 29, 14)` `mot` streams | `E104`, `E105` (`S12`, `ActionsEstLoader.py`, `pose_utils.py`) | `[SOURCE]` |
| `TwoStreamSpatialTemporalGraph` (10 `st_gcn` layers, symmetric $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, `tsstg-model.pth` 7-class Le2i weights) | `E101`, `E105`, `E110a`, `E111a`, `E111b` (`S1`, `S12`) | `[SOURCE]` |
| Optional secondary `PYSKL` (`S2`) 17-keypoint `ST-GCN` / `ST-GCN++` (`ntu60_xsub_hrnet/j.pth`, `b.pth`, Action `A43 = falling down`) plug-in backend | `E106`, `E109a`, `E109b` (`S2`, `kennymckormick/pyskl`) | `[SOURCE]` |
| Hysteresis run-merging (`hold_runs`, `merge_runs`) and per-person cooldown/realert deduplication | `E02`, `E03` (`S0`, `fall_monitor.py` L76-98, L145-170) | `[SOURCE]` |
| Replacing `S12`'s Tiny-YOLO + AlphaPose + SORT with `yolo26x-pose.pt` + `ByteTrack` to feed `TSSTG` | `E01`, `E102`, `E103` | `[UNTESTED DEPARTURE]` |
| Post-`scale_pose` aspect/confidence sanity guard on `TSSTG` `'Fall Down'` / `'Lying Down'` predictions | `E104`, `E108` | `[UNTESTED DEPARTURE]` |
| Prolonged unusual `immobility` detector (`detect_immobility`) using non-decaying reference height + keypoint motion threshold over `T_immobile` | `E02`, `E03` | `[UNTESTED DEPARTURE]` |

---

## 4. Components Deliberately Excluded (The Delete Pass)

1. **`S12`'s Legacy Detector (`Tiny-YOLO`), Pose Estimator (`AlphaPose / SPPE FastPose`), and Tracker (`SORT / DeepSORT`)**:
   - **Why Excluded**: `yolo26x-pose.pt` + `ByteTrack` (`S0`) already extracts 17 COCO keypoints and track IDs in a single unified pass (`E01`). Running a separate Tiny-YOLO + AlphaPose + SORT pipeline duplicates detection/pose/tracking compute for zero gain once the 17 COCO keypoints are sliced to the 13 `S12` keypoints (`E102`).
2. **`S0`'s Heuristic Fall & Lying Rule Trees (`detect_falls` drop/velocity/angle/aspect tree and `detect_lying` angle/aspect hysteresis)**:
   - **Why Excluded**: Per User Round 1 (`round-1.md`), the pretrained `TSSTG` (`tsstg-model.pth`) model directly outputs learned probabilities for `'Fall Down'` (`class 6`), `'Lying Down'` (`class 3`), `'Sitting'` (`class 2`), `'Sit down'` (`class 5`), `'Stand up'` (`class 4`), `'Standing'` (`class 0`), and `'Walking'` (`class 1`) trained on Le2i (`E101`).
3. **`PoseConv3D` (`SlowOnly-R50` 3D Heatmap Volume CNN from `S2`) in the Default Realtime Path**:
   - **Why Excluded**: Although `PoseConv3D` is more robust to limb keypoint dropout (`E108`), it requires rendering $17 \times 48 \times 56 \times 56$ Gaussian heatmaps and running a `15.9G` FLOPs 3D-CNN (`E107`), and its NTU-60 label map only includes `'falling down'` (`A43`, `E109a`) without a persistent `'Lying Down'` class. Kept only as an optional heavy benchmark comparator.
4. **33-Joint 3D MediaPipe BlazePose Hypergraph-Mamba (`S4` FLASH) & Unreleased GCNs (`S3`, `S6`, `S7`)**:
   - **Why Excluded**: No runnable model code or pretrained `.pth` weights exist in their repositories (`S3.md`, `S4.md`, `S6.md`, `S7.md`), and `S4` requires 33 3D BlazePose landmarks rather than 17 2D COCO keypoints.
5. **Hand-Crafted 4-Override Rule Equations from `S9` (Alam et al. 2026) and `S11` (Liu et al. 2025)**:
   - **Why Excluded**: Both sources had `read_status: PARTIAL` with unverified internal constants (`C9`), and `TSSTG` (`S12`) handles sitting/standing/walking separation via learned graph convolutions (`E101`).

---

## 5. Paper / Code Disagreements and How Each Is Resolved

1. **ST-GCN Layer Count (`9 layers` in `S1` Paper vs. `10 layers` in `S1` and `S12` Code — `E111a`)**:
   - **Disagreement**: `S1` (`arXiv:1801.07455`, Section 3.3) states 9 ST-GCN layers, whereas both `yysijie/st-gcn` (`net/st_gcn.py`) and `GajuuzZ/Human-Falling-Detect-Tracks` (`Actionsrecognition/Models.py`) instantiate **10 `st_gcn` layers** (Block 1 `in_channels -> 64` with `residual=False`, plus 9 residual layers).
   - **Resolution**: **Follow the code (10 layers).** `tsstg-model.pth` checkpoint state_dict keys (`pts_stream.st_gcn_networks.0..9`, `mot_stream.st_gcn_networks.0..9`) require all 10 layers to load without `RuntimeError`.
2. **Graph Adjacency Normalization (`normalize_digraph` in `S1` Code vs. `S12` Code — `E111b`)**:
   - **Disagreement**: `S1` Paper Eq. 3 specifies symmetric $\mathbf{\Lambda}^{-1/2}\mathbf{A}\mathbf{\Lambda}^{-1/2}$, `S1` Code (`yysijie/st-gcn/net/utils/graph.py`) implements asymmetric $\mathbf{A}\mathbf{D}^{-1}$, and `S12` Code (`Actionsrecognition/Utils.py`) implements symmetric $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ (`Dn[i, i] = Dl[i]**(-0.5); DAD = np.dot(np.dot(Dn, A), Dn)`).
   - **Resolution**: **Follow `S12` Code (`D^{-1/2} A D^{-1/2}`) for `tsstg-model.pth`**, because `tsstg-model.pth` was trained with `S12`'s symmetric `normalize_digraph`.

---

## 6. Faithfulness Declaration (`[UNTESTED DEPARTURE]` Flags)

Every departure from the exact source implementations is explicitly flagged below:

1. **`[UNTESTED DEPARTURE 1]` — Feeding `yolo26x-pose.pt` + `ByteTrack` Keypoints into `tsstg-model.pth` (Trained on `AlphaPose / FastPose` + `SORT`)**:
   - **Why**: The user requires building on top of `yolo26x-pose.pt` + `ByteTrack` (`S0`), whereas `S12` trained `tsstg-model.pth` on 13 COCO keypoints extracted by `SPPE FastPose (AlphaPose)` tracked with `SORT` at `stride=1` (`E101`, `E102`).
   - **Risk & Unvalidated Status**: `yolo26x-pose.pt` keypoint confidence distributions (`conf`) and frame stride (`stride=2` in `S0` vs. `stride=1` in `S12`) may shift the effective temporal velocity in `mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]` (`E105`). No source in the ledger validates `yolo26x-pose.pt` $\rightarrow$ `tsstg-model.pth` without testing frame stride (`stride=1` vs. `stride=2`).
2. **`[UNTESTED DEPARTURE 2]` — Handling Missing / Low-Confidence Keypoints (`NaN` vs. `0.0`) Before `scale_pose`**:
   - **Why**: In `S0` (`fall_monitor.py` L57), keypoints with `conf < 0.3` are masked to `np.nan`. In `S12` (`ActionsEstLoader.py` / `pose_utils.py`), `scale_pose` uses `np.nanmin` / `np.nanmax`, but `torch.tensor(pts)` passes straight into `nn.BatchNorm1d` and `Conv2d` where any surviving `NaN` poisons the entire 30-frame `TSSTG` forward pass (`NaN` output probabilities). Furthermore, `E108` (`S2`) proves coordinate GCNs lose `14.3` pp when a limb keypoint is dropped.
   - **Departure**: Forward-filling/interpolating short keypoint gaps along the 30-frame track buffer (or keeping `yolo26x-pose`'s raw `(x, y, conf)` coordinates when `conf > 0` instead of setting `NaN`) before `TSSTG.predict`.
3. **`[UNTESTED DEPARTURE 3]` — Post-`TSSTG` Aspect-Ratio / Bounding-Box Sanity Gate**:
   - **Why**: Because `S12`'s `scale_pose` (`E104`) stretches every frame's skeleton independently to `[-1, 1]^2`, a standing person with jittery keypoints can resemble a fallen skeleton in normalized `[-1, 1]^2` space. Combining `TSSTG`'s `'Fall Down'` / `'Lying Down'` probability with a lightweight bounding-box/torso check (e.g., excluding `'Lying Down'` when the person's bounding box is tall and narrow `aspect < 0.45` and torso is vertical `angle < 25°`) is a hybrid safeguard not present in `S12`.
4. **`[UNTESTED DEPARTURE 4]` — Prolonged Unusual Immobility Detector (`detect_immobility`)**:
   - **Why**: Required by Acceptance Criterion #3, but absent from both `S0` (`E03`) and `S12` (`E101`). Implemented by tracking mean keypoint displacement speed (`motion <= max_immobile_motion`) over a configured duration `T_immobile` while `TSSTG` classifies the person as stationary (`'Standing'`, `'Sitting'`, or `'Lying Down'`).

---

## 7. Unresolved `[UNKNOWN]`s for the Plan to Decide

1. **`[UNKNOWN 1]` — Frame Stride for `TSSTG` (`stride=1` vs. `stride=2`)**:
   - `S12` (`main.py`) runs `TSSTG` on every frame (`deque(maxlen=30)` at ~25 fps $\approx 1.2\text{s}$ window), whereas `S0` (`fall_monitor.py` L239) defaults to `--stride 2` (30 frames at stride 2 = $2.4\text{s}$ window, doubling inter-frame displacement `mot`).
2. **`[UNKNOWN 2]` — Probability Thresholds & Duration Constants (`p_fall_thresh`, `p_lying_thresh`, `T_lying`, `T_immobile`)**:
   - `S12` (`main.py`) simply takes `action_name = action_model.class_names[out[0].argmax()]` on every frame once `len(kp_list) == 30`, without a probability threshold or multi-second duration timer.
