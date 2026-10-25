# System Requirements, Acceptance Criteria & Phase-by-Phase Definition of DONE (`requirements.md`)

**Project:** Pose-Based Human Fall, Persistent Lying, and Prolonged Unusual Immobility Candidate Detection Pipeline  
**Companion Design:** [`research/design/design-3.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/design/design-3.md)  
**Companion Implementation Plan:** [`research/plan.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/plan.md)  
**Evidence Ledger:** [`research/ledger/evidence.jsonl`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/ledger/evidence.jsonl) & [`research/ledger/objections.jsonl`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/ledger/objections.jsonl)  
**Baseline Reference:** [`c:\Users\Maha Mamdouh\Downloads\fall_monitor.py`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py) (`S0`)

---

## 1. User Story & Scope Statement

> **User Story:**  
> *"As a health and safety manager, I want the system to identify potential falls, persons lying on the floor, and prolonged unusual immobility, so that situations requiring human assistance can be brought to an operator's attention."*

### In-Scope Deliverables
1. **Research & evaluation** of suitable pose, action-recognition, and temporal models (`S0`–`S14`, completed in `research/`).
2. **Fall-like event detection** (candidate `fall` alerts with high priority).
3. **Lying-person posture detection** (candidate `lying` alerts when floor posture persists for configured duration `T_lying`).
4. **Prolonged immobility detection** (candidate `immobility` alerts when a tracked person exhibits little observable movement over posture-configured durations `T_immobile_post_fall`, `T_immobile_lying`, `T_immobile_seated`).
5. **Duration thresholds, temporal logic, and structured event evidence** (keyframe snapshot, clip time window, 7-class `TSSTG` probabilities, kinematic & lower-limb metrics, and operator review lifecycle).
6. **Recorded-video testing and pipeline integration** (`fall_monitor.py` CLI and `evaluate_benchmark.py` held-out video evaluation).

---

## 2. Environment, Dependency, Asset & Licensing Requirements

### 2.1 Software & Package Requirements (`requirements.txt`)
| Requirement ID | Package / Runtime | Version Floor | Purpose & Evidence Locator |
|---|---|---|---|
| `ENV-1` | Python | `>= 3.10` | Type union syntax, `collections.deque`, `pathlib`, `dataclasses` |
| `ENV-2` | `ultralytics` | `>= 8.4.0` | Required for native `yolo26x-pose.pt` RLE pose weights and built-in `ByteTrack` (`FO2`, `E01`) |
| `ENV-3` | `lapx` | `>= 0.5.5` | Provides linear assignment solver `lap` required by `bytetrack.yaml` (`S0` L19) |
| `ENV-4` | `torch` | `>= 2.0.0` | `TwoStreamSpatialTemporalGraph` inference, `weights_only=True` checkpoint loading (`E105`, `FO2`) |
| `ENV-5` | `numpy` | `>= 1.24.0` | Keypoint slicing, `scale_pose`, `normalize_digraph`, 25 Hz interpolation (`E103`, `E104`) |
| `ENV-6` | `pandas` | `>= 2.0.0` | Time-indexed rolling smoothing (`_roll`) and benchmark report generation (`S0` L23, L48–51) |
| `ENV-7` | `opencv-python` | `>= 4.8.0` | Video capture, FPS inspection, skeleton/banner rendering, evidence keyframe saving (`S0` L19, L176–224) |
| `ENV-8` | `pyyaml` | `>= 6.0` | Loading threshold overrides (`load_cfg`, `S0` L23, L36–42) |
| `ENV-9` | `pytest` | `>= 7.4.0` | Running the 16 source-mapped unit, contract, divergence, and departure tests (`T1`–`T16`) |

### 2.2 Model Weights & Evaluation Data Requirements
| Asset ID | Asset Path / Source | Verification Requirement |
|---|---|---|
| `ASSET-1` | `yolo26x-pose.pt` (Ultralytics official asset) | Loaded via `YOLO("yolo26x-pose.pt")` (`ultralytics >= 8.4.0`); outputs `(N, 17, 3)` COCO keypoints (`E01`) |
| `ASSET-2` | `Models/TSSTG/tsstg-model.pth` (`GajuuzZ/Human-Falling-Detect-Tracks`, Google Drive `1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO` or verified mirror) | Must pass SHA-256 verification (`verify_sha256`), load with `torch.load(..., weights_only=True)`, and satisfy `torch.allclose(model.pts_stream.A, ckpt['pts_stream.A'], atol=1e-6)` **before** `load_state_dict(..., strict=True)` (`FO1`, `FO2`) |
| `ASSET-3` | **Held-Out Evaluation Videos** (Le2i **unseen** rooms `Office` and `Lecture room`, plus URFD / MCFD or recorded CCTV clips covering `bending`, `kneeling`, `sitting`, and `floor_work`) | Must **exclude** `Le2i/Coffee_room` and `Le2i/Home` from held-out accuracy claims because `tsstg-model.pth` was trained on `Coffee room` and `Home` (`E101`, `FO10`). Note: Le2i `.avi` files with broken MP3 tracks must be stripped via `ffmpeg -i in.avi -an -c:v libx264 -pix_fmt yuv420p out.mp4` (`S0` L17–18). |

### 2.3 Licensing & Governance Constraints (`FO2`)
- **`LIC-1` (Ultralytics)**: `ultralytics` is licensed under **AGPL-3.0**; commercial deployment requires open-sourcing the application under AGPL-3.0 or obtaining an Ultralytics Enterprise License.
- **`LIC-2` (Le2i Dataset & Pretrained `tsstg-model.pth`)**: The Le2i dataset is licensed under **CC-BY-NC-SA-3.0** (Non-Commercial Share-Alike), and `GajuuzZ/Human-Falling-Detect-Tracks` carries no explicit open-source license file. Any commercial deployment must re-train `TwoStreamSpatialTemporalGraph` (`finetune_tsstg_on_yolo26`) on commercially permitted or in-house skeleton datasets.

---

## 3. Functional & Non-Functional Requirements

### 3.1 Functional Requirements (`FR`)
| ID | Requirement Description | Source / Objection Mapping |
|---|---|---|
| `FR-1` | **Multi-Person Pose & Tracking Interface**: Preserve `S0`'s batch streaming signature `FallMonitor.update(t, ids, boxes, kps, frame=None)` and CLI `run(a)` (`--source`, `--model`, `--tsstg-weights`, `--tracker`, `--config`, `--stride`, `--imgsz`, `--conf-det`, `--device`, `--realert`, `--show`, `--save`, `--events`, `--evidence-dir`). | `E01`, `E05`, `FO9` |
| `FR-2` | **Spatial-Temporal Track Stitching**: When a new track ID `tid` appears within `stitch_s = 1.5s` of a recently lost track and its bounding-box center lies within `0.8 * lost.ref_scale`, transfer the lost track's state (`ref_scale`, `pending_fall_t0`, `post_fall_stable_s`, keypoint/kinematic history) to `tid` so mid-fall `ByteTrack` ID switches are not dropped. | `O5`, `FO6` |
| `FR-3` | **17-to-14 `coco_cut` Keypoint Adaptation & Finite Imputation**: Cut COCO eyes/ears (`[1, 2, 3, 4]`), keeping indices `[0, 5..16]` (`13` nodes), track boolean `is_confident` (`conf >= 0.3` and `(x, y) != (0, 0)`), impute missing/low-confidence coordinates via causal forward-fill (or frame-0 bbox center), scale each frame to `[-1, 1]` via `scale_pose` (`1e-6` floor), and append the 14th neck midpoint `(pts[:, 1, :] + pts[:, 2, :]) / 2.0`. | `E102`, `E103`, `E104`, `O2` |
| `FR-4` | **FPS-Aware 25 Hz Resampling, Gap Reset & Imputation-Jump Masking**: Resample each track's `[t - 1.16s, t]` window onto 30 uniform steps at `25.0 Hz` (`dt = 0.04s`); reset the 30-frame buffer if any tracking gap exceeds `hold_s = 1.0s`; and set `mot_tensor[:, :2, k, j] = 0.0` whenever `not (is_confident[k, j] and is_confident[k+1, j])` so forward-filled joints snapping back to real detections cannot inject false velocity spikes. | `E105`, `E105b`, `FO7` |
| `FR-5` | **Pre-State-Machine `YOLO26 -> TSSTG` Transfer Gate**: Before locking downstream rules, evaluate real `yolo26x-pose.pt` skeletons on held-out fall, sit, and look-alike clips across score modes (`raw`, `alphapose_calibrated`, `constant_0.9`), and provide `finetune_tsstg_on_yolo26()` if held-out zero-shot macro-F1 `< 0.80`. | `E101`, `E105`, `FO3` |
| `FR-6` | **Candidate `fall` Detection with Two-Phase Latch & Controlled-Action/Knee Veto**: Smooth `TSSTG` 7-class probabilities over `0.4s` (`min_action_score = 0.50`), compute lower-limb `knee_angle` (`[11,13,15]`, `[12,14,16]`), and veto both Path A (`TSSTG` class `6` + descent) and Path B (`S0` kinematic drop `drop > 0.10`, `vdown > 0.15` + `TSSTG` classes `3`/`6`) whenever `controlled_veto` holds (`probs_smooth[[0,1,2,4,5]]` dominates or controlled bend/kneel geometry). Confirm pending falls only when followed within `post_s = 3.0s` by `stable_s = 0.6s` of **low/horizontal lying posture** (`hip_rel < 0.45` and `(angle > 55 or aspect > 0.9 or head_rel < 0.0)`) with `motion <= 1.5` (never confirming on a seated person with `pred_idx == 6`). Keep Path C (`S0` kinematic detector) active when track duration `< 1.16s`. | `E01`, `E04`, `E05`, `E101`, `O5`, `O6`, `PO2`, `PO3`, `FO4` |
| `FR-7` | **Candidate `lying` Detection**: Emit a candidate `lying` event (`priority='normal'`) using `S0`'s `hold_runs(t, cond, hold_s=1.0)` and `merge_runs(runs, cooldown_s=5.0)` when lying posture (`lying_cond == 1.0` and `(not has_30 or pred_idx in (3, 6))`) persists for `>= T_lying` (`2.0s`) with mean motion `<= 1.0`. | `E04`, `E05`, `E101`, `FO11` |
| `FR-8` | **Candidate `immobility` Detection via Windowed Confident-Joint Excursion**: Compute `ref_scale` as the 90th percentile of a 60s rolling `hypot(w, h)` history clamped to `<= 1.15 * median` and floored by `ref_scale_init`. Over `immobile_win_s = 2.0s`, compute median max excursion `disp_win` strictly over joints with `>= 3` confident (`non-imputed`) frames. Emit a candidate `immobility` event (`priority='normal'`) when `disp_win <= 0.035` persists for `T_immobile_post_fall = 40.0s` (post-fall lying), `T_immobile_lying = 100.0s` (floor lying), or `T_immobile_seated = 150.0s` (seated/standing). (**Conservative thresholds authorized during Phase 3 implementation loop** — original design values 5s/10s/120s were superseded to reduce nuisance alerts on slow-recovery and resting subjects.) | `E02`, `E03`, `O3`, `O9`, `FO5` |
| `FR-9` | **Operator Review Lifecycle & Structured Event Evidence**: Assign every candidate event a unique `event_id` and `status = 'candidate'`, support `review_event(event_id, status, operator_note)` for `'acknowledged' | 'confirmed' | 'dismissed'`, deduplicate via `cooldown_s = 5.0s` and `realert_s = 60.0s` (`E05`), and attach an `evidence` payload (`snapshot_path`, `clip_window`, `tsstg_probs`, `kinematics`, `keypoint_summary`). | `E04`, `E05`, `FO9`, `FO11` |
| `FR-10` | **Held-Out Multi-Room & Look-Alike Evaluation Harness**: Parse Le2i ground-truth frame annotations, flag training-split leakage if `Coffee_room` or `Home` is passed, and evaluate on held-out `Office` / `Lecture_room` + look-alike clips (`bending`, `kneeling`, `sitting`, `floor_work`), reporting Precision, Recall, F1, False Alarms/Hour (`FA/h`), Mean Detection Latency (`s`), and Per-Activity False-Alarm Breakdown. | `E101`, `FO10` |

### 3.2 Non-Functional Requirements (`NFR`)
- **`NFR-1` (Numerical & Tensor Safety)**: `torch.isfinite(pts).all()` and `torch.isfinite(mot).all()` must hold on 100% of frames regardless of missing keypoints, zero-area bounding boxes, or single-point collapses (`O2`, `T6`, `T7`).
- **`NFR-2` (Bounded Per-Frame Latency)**: Kinematic rolling windows must be bounded (`<= 3.5s` for fall kinematics, `2.0s` for displacement excursion) so per-frame state updates execute in `< 2 ms` CPU time per tracked person (`PO4`).
- **`NFR-3` (Exact Upstream Graph Fidelity)**: `Graph(layout='coco_cut', strategy='spatial').A` must match `ckpt['pts_stream.A']` within `atol=1e-6` prior to weight loading (`FO1`, `T2`).
- **`NFR-4` (Deterministic Reproducibility)**: All unit and contract tests (`T1`–`T16`) must pass deterministically on CPU without network access once `tsstg-model.pth` is present.

---

## 4. Acceptance Criteria (`AC`) Traceability Matrix

| AC ID | Acceptance Criterion (from User Story Specification) | Governing Modules & Requirements | Verification Tests & Evaluation Script | Pass Criterion |
|---|---|---|---|---|
| **`AC-1`** | *Given a video sequence showing a fall-like event, when the selected model or temporal detection logic analyzes the sequence, then it can produce a candidate fall event.* | `tsstg_model.py`, `keypoint_adapter.py`, `fall_monitor.py` (`FR-2`, `FR-3`, `FR-4`, `FR-6`) | `T1`–`T4`, `T8`, `T11`, `T12`, and `evaluate_benchmark.py` (`T16`) | `T1`–`T4`, `T8`, `T11`, `T12` pass; candidate `fall` event (`priority='high'`, `status='candidate'`) emitted on fall sequences across Path A (`TSSTG`), Path B (`Hybrid`), Path C (`<30` frames), and stitched mid-fall ID switches. |
| **`AC-2`** | *Given a person lying on the floor, when the posture persists for the configured duration, then the system can produce a candidate lying-person event.* | `fall_monitor.py` (`detect_lying`, `hold_runs`, `merge_runs`, `FR-7`) | `T13` (`test_lying_candidate_hold_runs_and_merge_runs_semantics`) | No alert before `T_lying` (`2.0s`); emits candidate `lying` event (`priority='normal'`) once lying posture persists `>= 2.0s` (bridging gaps `<= hold_s = 1.0s` with `motion <= 1.0`). |
| **`AC-3`** | *Given a tracked person with little observable movement over a configured duration, when the immobility criteria are met, then the system can produce a candidate immobility event.* | `fall_monitor.py` (`update_ref_scale`, `compute_windowed_displacement`, `FR-8`) | `T10`, `T14` (`test_windowed_displacement_immobility_ignores_jitter_and_imputed_joints`) | `disp_win <= 0.035` on confident joints triggers candidate `immobility` at `5.0s` (post-fall), `10.0s` (lying), and `120.0s` (seated), resilient to 1.5–2.0 px 30 Hz pose jitter, 2-frame bbox merges, and forward-filled occluded joints. |
| **`AC-4`** | *The system distinguishes candidate detections from confirmed incidents.* | `fall_monitor.py` (`FallMonitor.update`, `FallMonitor.review_event`, `FR-9`) | `T15` (`test_candidate_lifecycle_dedup_event_evidence_and_cli_run`) | Every emitted alert has a unique `event_id`, `status == 'candidate'`, banner text `"... CANDIDATE - operator review"`, and transitions to `'acknowledged'`, `'confirmed'`, or `'dismissed'` only via explicit `review_event()` call. |
| **`AC-5`** | *The system handles common look-alike activities, including bending, kneeling, sitting, and ordinary work on the floor, in its evaluation.* | `fall_monitor.py` (`compute_knee_angle`, `controlled_veto`, strict lying post-fall stability, `FR-6`), `evaluate_benchmark.py` (`FR-10`) | `T12` (Sub-cases a & b), `T16`, and held-out look-alike video evaluation | Seated `pred_idx == 6` false positives and rapid sit-downs / controlled knee-bends (`Sitting`/`Sit down` + `knee_angle`) are vetoed (`0` false fall alerts in `T12a/b`), active floor work (`motion > 1.5` or `> 1.0`) is rejected, and `lookalike_fa_breakdown` documents performance per activity (`bending`, `kneeling`, `sitting`, `floor_work`). |
| **`AC-6`** | *The selected approach is evaluated on representative video sequences, with documented performance and limitations.* | `transfer_gate.py` (`FR-5`), `evaluate_benchmark.py` (`FR-10`) | `T9`, `T16`, and generated `evaluation_report.md` | Evaluates `FallMonitor` vs. `BaselineFallMonitor` on held-out Le2i rooms (`Office`, `Lecture room`) + look-alike clips, reporting Precision, Recall, F1, FA/h, Detection Latency (s), and explicit surviving limitations (`L1`, `L2`, licensing). |

---

## 5. Phase-by-Phase Implementation Structure & Definition of DONE (`DoD`)

Implementation is divided into **5 sequential phases** (matching Tasks 1–5 of [`plan.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/brain/4f76fa57-e597-4563-a470-8a8d7031890c/plan.md)). No phase may be marked `DONE` until every checkbox in its **Definition of DONE** is verified.

---

### Phase 8.1 (Task 1): Core Temporal Graph Model (`tsstg_model.py` & `Models/TSSTG/tsstg-model.pth`)
- **Objective**: Vendor the exact 14-node `coco_cut` graph and 10-layer `TwoStreamSpatialTemporalGraph` from `GajuuzZ/Human-Falling-Detect-Tracks` (`S12`) and verify lossless, secure loading of `tsstg-model.pth`.
- **Requirements Covered**: `ENV-4`, `ENV-5`, `ASSET-2`, `FR-5` (model core), `NFR-3`, `NFR-4`.
- **Files Created**:
  - `Models/TSSTG/tsstg-model.pth`
  - `tsstg_model.py`
  - `tests/test_tsstg_model.py`
- **Definition of DONE (`DoD-8.1`)**:
  - [ ] `Models/TSSTG/tsstg-model.pth` is downloaded, its SHA-256 hash is pinned in `TSSTG_WEIGHTS_SHA256`, and it loads via `torch.load(..., map_location=device, weights_only=True)`.
  - [ ] `Test T1` (`test_tsstg_ten_layers_matches_code_divergence`) passes: confirms 10 `st_gcn` layers per stream, layer 0 `residual == 0`, layers `4` and `7` temporal stride `(2, 1)`, and channels `[64, 64, 64, 64, 128, 128, 128, 256, 256, 256]`.
  - [ ] `Test T2` (`test_coco_cut_graph_matches_checkpoint_buffer_before_load`) passes: confirms `normalize_digraph` computes asymmetric $\mathbf{A}\mathbf{D}^{-1}$ and `torch.allclose(model.pts_stream.A, ckpt['pts_stream.A'], atol=1e-6)` **before** calling `load_state_dict`.
  - [ ] `Test T3` (`test_tsstg_pretrained_state_dict_loads_strictly_with_sha256`) passes: confirms `verify_sha256` and `load_state_dict(..., strict=True)` succeed with zero missing or unexpected keys.
  - [ ] `Test T4` (`test_tsstg_matches_upstream_golden_outputs`) passes: confirms numerical equivalence (`atol=1e-6`) between `TSSTGClassifier` and verbatim upstream `ActionsEstLoader.py::TSSTG.predict` on a fixed 30-frame skeleton fixture.

---

### Phase 8.2 (Task 2): 17-to-14 Keypoint Adapter, 25 Hz Resampling & Imputation-Jump Masking (`keypoint_adapter.py`)
- **Objective**: Convert variable-FPS, noisy 17-keypoint `yolo26x-pose.pt` tracks into finite, 25.0 Hz resampled `(1, 3, 30, 14)` `pts` and `(1, 2, 29, 14)` `mot` tensors without `(0, 0)` corner distortion, `NaN` propagation, or imputation-snap velocity spikes.
- **Requirements Covered**: `FR-3`, `FR-4`, `NFR-1`, `NFR-4`.
- **Files Created**:
  - `keypoint_adapter.py`
  - `tests/test_keypoint_adapter.py`
- **Definition of DONE (`DoD-8.2`)**:
  - [ ] `Test T5` (`test_17_to_13_keypoint_slice_and_14th_neck_node`) passes: verifies eye/ear removal (`[0, 5..16]`) and neck midpoint node `13` across `(x, y, score)`.
  - [ ] `Test T6` (`test_finite_imputation_and_imputation_jump_masking_in_mot`) passes: verifies causal forward-fill / frame-0 bbox-center fallback, `torch.isfinite(pts).all()`, `torch.isfinite(mot).all()`, and `mot[:, :2, k, j] == 0.0` when an imputed joint snaps back to a real detection.
  - [ ] `Test T7` (`test_scale_pose_per_frame_range_and_zero_span_guard`) passes: verifies per-frame `[-1, 1]` scaling, `1e-6` zero-span protection, and `(width, height)` linear scale invariance.
  - [ ] `Test T8` (`test_fps_resampling_to_25hz_and_gap_reset_policy`) passes: verifies that `12.5 Hz` and `25.0 Hz` samplings of the same physical motion produce identical `pts` and `mot` tensors (`atol=1e-4`), and that any tracking gap `> hold_s` (`1.0s`) returns `None` (resetting the buffer).
  - [ ] Full test suite `pytest` runs green (`T1`–`T8`).

---

### Phase 8.3 (Task 3): Pre-State-Machine `YOLO26 -> TSSTG` Transfer & Calibration Gate (`transfer_gate.py`)
- **Objective**: Directly test `[UNTESTED DEPARTURE 1]` (whether `tsstg-model.pth` trained on AlphaPose heatmaps + SORT transfers accurately to `yolo26x-pose.pt` RLE keypoints + ByteTrack on held-out sequences) before building the downstream state machine.
- **Requirements Covered**: `FR-5`, `AC-6`.
- **Files Created**:
  - `transfer_gate.py`
  - `tests/test_transfer_gate.py`
- **Definition of DONE (`DoD-8.3`)**:
  - [ ] `Test T9` (`test_transfer_gate_evaluates_score_modes_and_finetune_step`) passes: verifies `evaluate_tsstg_transfer_on_sequences` across `('raw', 'alphapose_calibrated', 'constant_0.9')` and verifies that `finetune_tsstg_on_yolo26` reduces `BCELoss` over 3 gradient steps while keeping graph buffer `A` invariant.
  - [ ] **Transfer Gate Decision Recorded**: Run `transfer_gate.py` on held-out fall, sitting, and floor-work sequences extracted with `yolo26x-pose.pt`:
    - If zero-shot transfer macro-F1 `>= 0.80` under the best `score_mode`, lock `THRESHOLDS['sampling']['score_mode']` to that mode.
    - If zero-shot transfer macro-F1 `< 0.80`, execute `finetune_tsstg_on_yolo26()` on `yolo26x-pose.pt` extracted training sequences and save the calibrated checkpoint before proceeding to Phase 8.4.
  - [ ] Full test suite `pytest` runs green (`T1`–`T9`).

---

### Phase 8.4 (Task 4): Hybrid State Machine (`fall`, `lying`, `immobility`), Track Stitching, Windowed Excursion & Event Evidence (`fall_monitor.py`)
- **Objective**: Implement the upgraded `FallMonitor` (preserving `BaselineFallMonitor` and `S0`'s CLI/batch interface) with track stitching, `0.4s`-smoothed `TSSTG` + `knee_angle` controlled-action veto, strict lying post-fall stability, 2.0s windowed confident-joint excursion for immobility, operator review lifecycle, and structured event evidence.
- **Requirements Covered**: `FR-1`, `FR-2`, `FR-6`, `FR-7`, `FR-8`, `FR-9`, `NFR-2`, `AC-1`, `AC-2`, `AC-3`, `AC-4`, `AC-5`.
- **Files Created**:
  - `fall_monitor.py`
  - `tests/test_fall_monitor.py`
- **Definition of DONE (`DoD-8.4`)**:
  - [ ] `Test T10` (`test_ref_scale_outlier_clamped_and_non_decaying`) passes: confirms orientation invariance, non-decay over `>20s` of compressed sitting, and outlier clamping (`<= 1.15 * median`) during 2-frame multi-person bbox merges.
  - [ ] `Test T11` (`test_track_stitching_and_cold_start_under_30_frames`) passes: confirms mid-fall ID switch stitching (`tid=1 -> tid=2`) and `< 30` frame cold-start kinematic fallback (`Path C`).
  - [ ] `Test T12` (`test_hybrid_fall_latch_requires_descent_and_lying_stability_and_vetoes_seated`) passes: confirms (a) seated `pred_idx == 6` false alarm rejection, (b) rapid sit-down / knee-bend veto, (c) Path A `TSSTG` fall + `0.6s` lying stability confirmation, and (d) Path B hybrid kinematic catch when class `6` is skipped.
  - [ ] `Test T13` (`test_lying_candidate_hold_runs_and_merge_runs_semantics`) passes: confirms `hold_runs` and `merge_runs` bridge a `0.5s <= hold_s` gap to emit `lying` at `2.0s`.
  - [ ] `Test T14` (`test_windowed_displacement_immobility_ignores_jitter_and_imputed_joints`) passes: confirms 30 Hz 1.5 px jitter resilience, exclusion of forward-filled joints, and differentiated triggers at `5.0s` (post-fall), `10.0s` (lying), and `120.0s` (seated).
  - [ ] `Test T15` (`test_candidate_lifecycle_dedup_event_evidence_and_cli_run`) passes: confirms `event_id`, `status == 'candidate'`, `review_event()` state transitions (`acknowledged`, `confirmed`, `dismissed`), `cooldown_s` / `realert_s` deduplication, saved keyframe snapshot + `evidence` dict, and end-to-end `run()` video processing.
  - [ ] Full test suite `pytest` runs green (`T1`–`T15`).

---

### Phase 8.5 (Task 5): Held-Out Multi-Room & Look-Alike Video Evaluation (`evaluate_benchmark.py`)
- **Objective**: Evaluate `FallMonitor` vs. `BaselineFallMonitor` on held-out video sequences (unseen Le2i rooms `Office` and `Lecture room`, plus look-alike activities `bending`, `kneeling`, `sitting`, `floor_work`) with ground-truth annotations and produce the final documented performance and limitations report.
- **Requirements Covered**: `FR-10`, `ASSET-3`, `LIC-1`, `LIC-2`, `AC-5`, `AC-6`.
- **Files Created**:
  - `evaluate_benchmark.py`
  - `tests/test_evaluate_benchmark.py`
  - `evaluation_report.md` (generated output report)
- **Definition of DONE (`DoD-8.5` & Project-Wide Final DoD)**:
  - [ ] `Test T16` (`test_le2i_annotation_parser_and_heldout_benchmark_metrics`) passes: confirms accurate parsing of Le2i frame annotations, calculation of `precision`, `recall`, `f1`, `false_alarms_per_hour`, `mean_detection_latency_s`, `lookalike_fa_breakdown`, and automatic leakage warning if `Coffee_room` or `Home` paths are included.
  - [ ] All 16 automated tests (`T1`–`T16`) pass in a single bare `pytest` run with 0 failures and 0 errors.
  - [ ] `evaluate_benchmark.py` is executed on the representative held-out video manifest, producing `evaluation_report.md` with side-by-side tables comparing `BaselineFallMonitor` (`S0`) vs. upgraded `FallMonitor` across:
    1. Fall Detection Precision, Recall, F1, and Mean Detection Latency (`s`),
    2. Persistent Lying Detection Recall & Precision,
    3. Prolonged Immobility Detection Recall & False Alarms per Hour (`FA/h`),
    4. Per-Activity Look-Alike False Alarm Counts (`bending`, `kneeling`, `sitting`, `floor_work`),
    5. Documented surviving limitations (`Limitation L1` on extreme occlusion/unseen floor postures, `Limitation L2` on seated stillness threshold tuning, and `LIC-1`/`LIC-2` commercial licensing requirements).
