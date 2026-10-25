# Round 1 (User Round Protocol)

## Your suggestion
> **A1 (Phase 4 Exit)**: "Pure Priority 1 (`Option 2` from last round): Center the design directly around integrating the pretrained `TSSTG` (`tsstg-model.pth`) or `PYSKL` (`S2`) temporal model on top of `yolo26x-pose.pt` + `ByteTrack`, using lightweight rules only for duration/immobility tracking"
> **A2 (Kinematic Upgrades on `fall_monitor.py`)**: "we wont need this file anymore if our design using the pretrained weights (the temporal open weights you found) works"

## Classification
- **A1 (Centering design around pretrained `TSSTG` / `PYSKL` temporal model on top of `yolo26x-pose.pt` + `ByteTrack`, using lightweight rules only for duration/immobility tracking)**: **Correct.**
- **A2 ("we wont need this file [`fall_monitor.py`] anymore if our design using the pretrained weights works")**: **Partially correct.**
  - **What holds**: We do **not** need `fall_monitor.py`'s complicated heuristic fall-drop (`detect_falls`) or lying-angle (`detect_lying`) rule trees once the pretrained `TSSTG` (`tsstg-model.pth`) 7-class temporal model is classifying 30-frame windows directly into `['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']` (`E101`).
  - **What fails**: Neither `TSSTG` (`S12`) nor `PYSKL` (`S2`) can replace the **duration persistence timers, cooldown/deduplication state machine, or prolonged immobility displacement tracker** required by your Acceptance Criteria #2, #3, and #4. Specifically:
    1. `TSSTG` only classifies a fixed **30-frame window** (~1.2s at 25 fps) into instantaneous action probabilities (`E101`, `E105`). It has no internal memory across multiple seconds to verify whether `Lying Down` persists for a configured duration $T_{\text{lying}}$ (AC #2).
    2. `TSSTG`'s 7 classes include `'Standing'`, `'Sitting'`, and `'Lying Down'` (`E101`), which fire for *every* normal standing or seated person. To detect **prolonged unusual immobility** (AC #3: *"Given a tracked person with little observable movement over a configured duration"*), the pipeline still requires a lightweight keypoint displacement tracker over the configured duration $T_{\text{immobile}}$.
    3. When keypoints are occluded ($p=1$ limb dropout), coordinate-based GCNs like `TSSTG` lose significant accuracy (`-14.3` percentage points in `E108`), and `TSSTG`'s `scale_pose` stretches any bounding box to `[-1, 1]^2` (`E104`), meaning a single-frame confidence gate (`conf_kp`) and temporal persistence filter over `TSSTG` probabilities are still needed to avoid false candidate alerts.

## What the sources show
- `[SOURCE]` `TSSTG` (`tsstg-model.pth`) classifies 30-frame 14-node `coco_cut` sequences into 7 classes: `['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']` — `S12, ActionsEstLoader.py, class TSSTG.__init__ (E101)`.
- `[SOURCE]` `TSSTG.predict` slices 13 keypoints from the 17 COCO keypoints (dropping eyes and ears `1, 2, 3, 4` via `pose_hm[:, :1], pose_hm[:, 5:]`) and appends the midpoint of Left/Right Shoulders (`(pts[:, 1, :] + pts[:, 2, :]) / 2`) as node 13, taking `pts` of shape `(1, 3, 30, 14)` and `mot` of shape `(1, 2, 29, 14)` — `S12, PoseEstimateLoader.py & ActionsEstLoader.py (E102, E103, E105)`.
- `[SOURCE]` `TSSTG.predict` applies `scale_pose`, scaling each frame's `xy` independently to `[-1, 1]^2` via `((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1` — `S12, pose_utils.py (E104)`.
- `[SOURCE]` `PYSKL` (`S2`) provides pretrained 17-COCO-keypoint `ST-GCN`, `ST-GCN++`, and `PoseConv3D` checkpoints on NTU-RGB+D 60/120 where class index 42 (`Action A43`) is `'falling down'`, and demonstrates that coordinate GCNs drop by `14.3` percentage points under limb keypoint dropout ($p=1$) compared to `0.9` percentage points for `PoseConv3D` — `S2, arXiv:2104.13586 Table 4 & configs/stgcn/README.md (E106, E107, E108, E109a)`.
- `[UNKNOWN]` Neither `S12` (`TSSTG`) nor `S2` (`PYSKL`) defines a duration threshold or classifier class for "prolonged unusual immobility" (`C9`).

## What changed in the design
- **Adopted Pure Priority 1 (`TSSTG` / `PYSKL` Pretrained Temporal Model Core)**: Replaced `fall_monitor.py`'s heuristic `detect_falls` and `detect_lying` rule trees with a clean, modular **Pretrained Skeleton Action Recognition Pipeline** centered on `yolo26x-pose.pt` + `ByteTrack` $\rightarrow$ 30-frame per-track keypoint buffer $\rightarrow$ pretrained `TSSTG` (`tsstg-model.pth`, primary 7-class Le2i model) with optional `PYSKL` (`ST-GCN++` / `PoseConv3D` `A43` fall verifier) $\rightarrow$ lightweight Temporal Persistence & Immobility State Machine.
- **Dropped Heuristic Angle/Aspect Rule Trees**: Removed `fall_monitor.py`'s `detect_falls`, `detect_lying`, `lying_cond`, and lower-limb heuristic angle overrides from the core classifier, keeping only the minimal temporal duration/cooldown state machine and keypoint-motion immobility accumulator needed for AC #2, AC #3, and AC #4.

## Open objections
- To be generated and resolved in the Phase 6 Critic Subagent pass (`rounds/round-1-critic.md` and `ledger/objections.jsonl`).
