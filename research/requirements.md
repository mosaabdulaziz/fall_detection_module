# System Requirements, Acceptance Criteria & Phase-by-Phase Definition of DONE (`requirements.md`)

**Project:** Pose-Based Human Fall, Persistent Lying, and Prolonged Unusual Immobility Candidate Detection Pipeline (OpenMMLab MMAction2 / PYSKL 17-COCO-Keypoint Multi-Stream ST-GCN + `yolo26x-pose.pt`)  
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
2. **Fall-like event detection** (candidate `fall` alerts with high priority, powered by OpenMMLab MMAction2 / PYSKL 17-COCO-keypoint Multi-Stream ST-GCN `A43: falling` + hybrid kinematic latch).
3. **Lying-person posture detection** (candidate `lying` alerts when floor posture persists for configured duration `T_lying`).
4. **Prolonged immobility detection** (candidate `immobility` alerts when a tracked person exhibits little observable movement over posture-configured durations `T_immobile_post_fall`, `T_immobile_lying`, `T_immobile_seated`).
5. **Duration thresholds, temporal logic, and structured event evidence** (keyframe snapshot, clip time window, 7-class projected & top-5 NTU-60 `MMAction2TSSTG` probabilities, kinematic & lower-limb metrics, and operator review lifecycle).
6. **Recorded-video testing and pipeline integration** (`fall_monitor.py` CLI and `evaluate_benchmark.py` held-out video evaluation).

---

## 2. Environment, Dependency, Asset & Licensing Requirements

### 2.1 Software & Package Requirements (`F:\fall_detection\.venv`)
| Requirement ID | Package / Runtime | Version Floor | Purpose & Evidence Locator |
|---|---|---|---|
| `ENV-1` | Python | `>= 3.10` | Type union syntax, `collections.deque`, `pathlib`, `dataclasses` (`F:\fall_detection\.venv`) |
| `ENV-2` | `ultralytics` | `>= 8.4.0` | Required for native `yolo26x-pose.pt` RLE pose weights and built-in `ByteTrack` (`FO2`, `E01`) |
| `ENV-3` | `lapx` | `>= 0.5.5` | Provides linear assignment solver `lap` required by `bytetrack.yaml` (`S0` L19) |
| `ENV-4` | `torch` | `>= 2.0.0` | Standalone pure-PyTorch OpenMMLab `RecognizerGCN` (`STGCN` + `GCNHead`) inference and `weights_only=True` checkpoint loading (`E203`, `FO2`) — **no `mmcv` or `mmengine` runtime dependency required** |
| `ENV-5` | `numpy` | `>= 1.24.0` | Native 17-COCO-keypoint `PreNormalize2D`, `GenSkeFeat` (`j`, `jm`, `b`, `bm`), `UniformSampleFrames(100)`, `FormatGCNInput(M=2)`, `normalize_digraph`, 25 Hz interpolation (`E202`, `E204`) |
| `ENV-6` | `pandas` | `>= 2.0.0` | Time-indexed rolling smoothing (`_roll`) and benchmark report generation (`S0` L23, L48–51) |
| `ENV-7` | `opencv-python` | `>= 4.8.0` | Video capture, FPS inspection, skeleton/banner rendering, evidence keyframe saving (`S0` L19, L176–224) |
| `ENV-8` | `pyyaml` | `>= 6.0` | Loading threshold overrides (`load_cfg`, `S0` L23, L36–42) |
| `ENV-9` | `pytest` | `>= 7.4.0` | Running the 16 source-mapped unit, contract, divergence, and departure tests (`T1`–`T16`) |

### 2.2 Model Weights & Evaluation Data Requirements (`F:` Drive Storage Rule)
- **Storage Rule (`STORAGE-1`)**: All downloaded model checkpoints and evaluation video assets MUST be stored on the `F:` drive (`F:\fall_detection\weights\`); never download model weights onto the `C:` drive.

| Asset ID | Asset Path / Official OpenMMLab Source (`E201`) | Verification Requirement |
|---|---|---|
| `ASSET-1` | `yolo26x-pose.pt` (Ultralytics official asset on `F:\fall_detection\weights\`) | Loaded via `YOLO` (`ultralytics >= 8.4.0`); outputs native `(N, 17, 3)` COCO keypoints (`E01`) |
| `ASSET-2A` | **OpenMMLab MMAction2 v1.0 17-COCO-Keypoint ST-GCN Checkpoints** (`F:\fall_detection\weights\`, `12,479,895` bytes each, Apache-2.0):<br>• `mmaction2_joint.pth` (`sha256: 484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea`)<br>• `mmaction2_bone.pth` (`sha256: c4b44488123d68e19d742ade26d02ff638a911bb0084fcfa7b54d751957318dd`)<br>• `mmaction2_jm.pth` (`sha256: f18eb408a42eebced02fc0e0dfeed2c6edb320cc1752abc6df855dff0845fc6d`)<br>• `mmaction2_bm.pth` (`sha256: 99c60e2d462a3978114193f35b5fce465bfa32de2275325913ab8a394900bfb0`) | Must pass SHA-256 verification (`verify_sha256`), load with `torch.load(..., weights_only=True)`, and satisfy `torch.allclose(model.backbone.gcn[0].gcn.A, sd['backbone.gcn.0.gcn.A'], atol=1e-6)` **before** `load_state_dict(..., strict=True)` (`E201`–`E203`, `FO1`, `FO2`) |
| `ASSET-2B` | **OpenMMLab PYSKL HRNet 17-COCO-Keypoint ST-GCN Checkpoints** (`F:\fall_detection\weights\`, `12,474,519` bytes each, Apache-2.0):<br>• `pyskl_j.pth` (`sha256: b4e476a34f2fe641f111cf8b8b66d08fca163a0b45e3df6b9ebb5c93ac9a71b2`)<br>• `pyskl_b.pth` (`sha256: e9d7cae55964837e6bd9ba5b00f346655da1937317421d79a8963fa9cf0c7fed`)<br>• `pyskl_jm.pth` (`sha256: 131738564797c3d1c4fc5f3580e4cda0303cffa16efb5f46f4c924e6d9488615`)<br>• `pyskl_bm.pth` (`sha256: 0e55b74c7d50410ec64ddb4527be6c83145ced62ed671383df45ccd1a2225ac4`) | Automatically remaps `cls_head.fc_cls.{weight,bias} -> cls_head.fc.{weight,bias}` and loads with `strict=True` (`E203`) |
| `ASSET-3` | **Held-Out Evaluation Videos** (Le2i rooms `Office`, `Lecture room`, `Coffee room`, `Home` — all 100% unseen by NTU-60 trained MMAction2/PYSKL weights — plus URFD / MCFD or recorded CCTV clips covering `bending`, `kneeling`, `sitting`, and `floor_work`) | Because MMAction2/PYSKL checkpoints were trained on NTU-RGB+D 60 (`E201`), there is zero training-split leakage on Le2i (`FO10`). Note: Le2i `.avi` files with broken MP3 tracks must be stripped via `ffmpeg -i in.avi -an -c:v libx264 -pix_fmt yuv420p out.mp4` (`S0` L17–18). |

### 2.3 Licensing & Governance Constraints (`FO2`, `E201`)
- **`LIC-1` (OpenMMLab MMAction2 & PYSKL)**: Code and checkpoints are licensed under **Apache-2.0** (permissive commercial and research license, resolving the unlicensed-repo issue of `GajuuzZ/Human-Falling-Detect-Tracks`).
- **`LIC-2` (Ultralytics)**: `ultralytics` is licensed under **AGPL-3.0**; commercial deployment requires open-sourcing the application under AGPL-3.0 or obtaining an Ultralytics Enterprise License.
- **`LIC-3` (Datasets)**: The Le2i dataset is licensed under **CC-BY-NC-SA-3.0** and NTU-RGB+D 60 has academic research terms; commercial deployment requires verifying dataset terms or fine-tuning on commercially licensed/in-house CCTV skeleton data.

---

## 3. Functional & Non-Functional Requirements

### 3.1 Functional Requirements (`FR`)
| ID | Requirement Description | Source / Objection Mapping |
|---|---|---|
| `FR-1` | **Multi-Person Pose & Tracking Interface**: Preserve `S0`'s batch streaming signature `FallMonitor.update(t, ids, boxes, kps, frame=None)` and CLI `run(a)` (`--source`, `--model`, `--weight-dir`, `--variant`, `--streams`, `--tracker`, `--config`, `--stride`, `--imgsz`, `--conf-det`, `--device`, `--realert`, `--show`, `--save`, `--events`, `--evidence-dir`). | `E01`, `E05`, `FO9` |
| `FR-2` | **Spatial-Temporal Track Stitching**: When a new track ID `tid` appears within `stitch_s = 1.5s` of a recently lost track and its bounding-box center lies within `0.8 * lost.ref_scale`, transfer the lost track's state (`ref_scale`, `pending_fall_t0`, `post_fall_stable_s`, keypoint/kinematic history) to `tid` so mid-fall `ByteTrack` ID switches are not dropped. | `O5`, `FO6` |
| `FR-3` | **Native 17-COCO-Keypoint Sanitization, `PreNormalize2D` & `GenSkeFeat`**: Retain all 17 native COCO keypoints (`0..16`, `V=17`), track boolean `is_confident` (`conf >= 0.3` and `(x, y) != (0, 0)`), impute missing/low-confidence coordinates via causal forward-fill (or frame-0 bbox center), apply MMAction2 `PreNormalize2D` (`(x - w/2)/(w/2)`, `(y - h/2)/(h/2)`), and compute `GenSkeFeat(dataset='coco', feats=['j', 'jm', 'b', 'bm'])` **before** temporal sampling to `clip_len=100`. | `E202`, `E204`, `O2` |
| `FR-4` | **FPS-Aware 25 Hz Resampling, Gap Reset, Imputation-Jump Masking, `UniformSampleFrames(100)` & `FormatGCNInput(M=2)`**: Resample each track's `[t - 1.16s, t]` window onto 30 uniform steps at `25.0 Hz` (`dt = 0.04s`); reset the buffer if any tracking gap exceeds `hold_s = 1.0s`; set `jm[:, k, v, :2] = 0.0` (and connected `bm`) whenever `not (is_confident[k, v] and is_confident[k+1, v])`; sample `T=30 -> 100` via `np.mod(np.arange(100), T)`; and zero-pad `M=1 -> M=2` (`(1, 2, 100, 17, 3)`). | `E204`, `FO7` |
| `FR-5` | **Standalone OpenMMLab `MMAction2TSSTG` (`RecognizerGCN`) & Pre-State-Machine Transfer Gate**: Implement zero-`mmcv`/zero-`mmengine` `RecognizerGCN` (`Graph(layout='coco', mode='stgcn_spatial')`, 10-stage `STGCN`, `GCNHead`), project 60 NTU-60 classes (`42: falling`, `7: sitting down`, `8: standing up`, `5: pickup`, `15/16: shoe kneeling`) to the 7 canonical classes (`probs_7`), evaluate `yolo26x-pose.pt` transfer across `prenormalize2d` vs. `track_compact`, and provide `finetune_tsstg_on_yolo26()`. | `E201`–`E205`, `FO1`–`FO3` |
| `FR-6` | **Candidate `fall` Detection with Two-Phase Latch & Controlled-Action/Knee Veto**: Smooth `probs_7` over `0.4s` (`min_action_score = 0.50`), compute lower-limb `knee_angle` (`[11,13,15]`, `[12,14,16]`), and veto both Path A (`MMAction2TSSTG` class `6` / `A43: falling` + descent) and Path B (`S0` kinematic drop `drop > 0.10`, `vdown > 0.15` + `probs_smooth[3] + probs_smooth[6] >= 0.50`) whenever `controlled_veto` holds (`probs_smooth[[0,1,2,4,5]]` dominates — including `A8: sitting down`, `A6: pickup`, `A16/17: shoe` — or controlled bend/kneel geometry). Confirm pending falls only when followed within `post_s = 3.0s` by `stable_s = 0.6s` of **low/horizontal lying posture** (`hip_rel < 0.45` and `(angle > 55 or aspect > 0.9 or head_rel < 0.0)`) with `motion <= 1.5` (never confirming on a seated person). Keep Path C (`S0` kinematic detector) active when track duration `< 1.16s`. | `E01`, `E04`, `E05`, `E205`, `O5`, `O6`, `PO2`, `PO3`, `FO4` |
| `FR-7` | **Candidate `lying` Detection**: Emit a candidate `lying` event (`priority='normal'`) using `S0`'s `hold_runs(t, cond, hold_s=1.0)` and `merge_runs(runs, cooldown_s=5.0)` when lying posture (`lying_cond == 1.0` and `(not has_30 or not controlled_veto)`) persists for `>= T_lying` (`2.0s`) with mean motion `<= 1.0`. | `E04`, `E05`, `E205`, `FO11` |
| `FR-8` | **Candidate `immobility` Detection via Windowed Confident-Joint Excursion**: Compute `ref_scale` as the 90th percentile of a 60s rolling `hypot(w, h)` history clamped to `<= 1.15 * median` and floored by `ref_scale_init`. Over `immobile_win_s = 2.0s`, compute median max excursion `disp_win` strictly over joints $j \in \{0,\dots,16\}$ with `>= 3` confident (`non-imputed`) frames. Emit a candidate `immobility` event (`priority='normal'`) when `disp_win <= 0.035` persists for `T_immobile_post_fall = 40.0s` (post-fall lying), `T_immobile_lying = 100.0s` (floor lying), or `T_immobile_seated = 150.0s` (seated/standing). (**Conservative thresholds authorized during Phase 3 implementation loop** — original design values 5s/10s/120s were superseded to reduce nuisance alerts on slow-recovery and resting subjects.) | `E02`, `E03`, `O3`, `O9`, `FO5` |
| `FR-9` | **Operator Review Lifecycle & Structured Event Evidence**: Assign every candidate event a unique `event_id` and `status = 'candidate'`, support `review_event(event_id, status, operator_note)` for `'acknowledged' | 'confirmed' | 'dismissed'`, deduplicate via `cooldown_s = 5.0s` and `realert_s = 60.0s` (`E05`), and attach an `evidence` payload (`snapshot_path`, `clip_window`, `tsstg_probs`, `ntu60_top5`, `kinematics`, `keypoint_summary`). | `E04`, `E05`, `FO9`, `FO11` |
| `FR-10` | **Held-Out Multi-Room & Look-Alike Evaluation Harness**: Parse Le2i ground-truth frame annotations and evaluate on held-out Le2i + look-alike clips (`bending`, `kneeling`, `sitting`, `floor_work`), reporting Precision, Recall, F1, False Alarms/Hour (`FA/h`), Mean Detection Latency (`s`), and Per-Activity False-Alarm Breakdown. | `E201`, `E205`, `FO10` |

### 3.2 Non-Functional Requirements (`NFR`)
- **`NFR-1` (Numerical & Tensor Safety)**: `torch.isfinite(tensor).all()` must hold on 100% of frames across all streams (`j`, `jm`, `b`, `bm`) regardless of missing keypoints, zero-area bounding boxes, or single-point collapses (`O2`, `T6`, `T7`).
- **`NFR-2` (Bounded Per-Frame Latency)**: Kinematic rolling windows must be bounded (`<= 3.5s` for fall kinematics, `2.0s` for displacement excursion) so per-frame state updates execute in `< 2 ms` CPU time per tracked person (`PO4`).
- **`NFR-3` (Exact Upstream Graph Fidelity)**: `Graph(layout='coco', mode='stgcn_spatial').A` (`(3, 17, 17)`) must match `sd['backbone.gcn.0.gcn.A']` within `atol=1e-6` prior to weight loading across all 8 MMAction2 and PYSKL checkpoints (`E202`, `FO1`, `T2`).
- **`NFR-4` (Deterministic Reproducibility & Storage Discipline)**: All unit and contract tests (`T1`–`T16`) must pass deterministically using `F:\fall_detection\.venv` and `F:\fall_detection\weights` without downloading any weights onto the `C:` drive.

---

## 4. Acceptance Criteria (`AC`) Traceability Matrix

| AC ID | Acceptance Criterion (from User Story Specification) | Governing Modules & Requirements | Verification Tests & Evaluation Script | Pass Criterion |
|---|---|---|---|---|
| **`AC-1`** | *Given a video sequence showing a fall-like event, when the selected model or temporal detection logic analyzes the sequence, then it can produce a candidate fall event.* | `tsstg_model.py`, `keypoint_adapter.py`, `fall_monitor.py` (`FR-2`, `FR-3`, `FR-4`, `FR-5`, `FR-6`) | `T1`–`T4`, `T8`, `T11`, `T12`, and `evaluate_benchmark.py` (`T16`) | `T1`–`T4`, `T8`, `T11`, `T12` pass; real `MMAction2TSSTG` predicts `probs_7[6] > 0.90` (`A43: falling`) on fall sequences (`T4`); candidate `fall` event (`priority='high'`, `status='candidate'`) emitted across Path A, Path B, Path C, and stitched ID switches. |
| **`AC-2`** | *Given a person lying on the floor, when the posture persists for the configured duration, then the system can produce a candidate lying-person event.* | `fall_monitor.py` (`detect_lying`, `hold_runs`, `merge_runs`, `FR-7`) | `T13` (`test_lying_candidate_hold_runs_and_merge_runs_semantics`) | No alert before `T_lying` (`2.0s`); emits candidate `lying` event (`priority='normal'`) once lying posture persists `>= 2.0s` (bridging gaps `<= hold_s = 1.0s` with `motion <= 1.0`). |
| **`AC-3`** | *Given a tracked person with little observable movement over a configured duration, when the immobility criteria are met, then the system can produce a candidate immobility event.* | `fall_monitor.py` (`update_ref_scale`, `compute_windowed_displacement`, `FR-8`) | `T10`, `T14` (`test_windowed_displacement_immobility_ignores_jitter_and_imputed_joints`) | `disp_win <= 0.035` on confident joints triggers candidate `immobility` at `5.0s` (post-fall), `10.0s` (lying), and `120.0s` (seated), resilient to 1.5 px 25–30 Hz pose jitter, 2-frame bbox merges, and forward-filled occluded joints. |
| **`AC-4`** | *The system distinguishes candidate detections from confirmed incidents.* | `fall_monitor.py` (`FallMonitor.update`, `FallMonitor.review_event`, `FR-9`) | `T15` (`test_candidate_lifecycle_dedup_event_evidence_and_cli_run`) | Every emitted alert has a unique `event_id`, `status == 'candidate'`, banner text `"... CANDIDATE - operator review"`, and transitions to `'acknowledged'`, `'confirmed'`, or `'dismissed'` only via explicit `review_event()` call. |
| **`AC-5`** | *The system handles common look-alike activities, including bending, kneeling, sitting, and ordinary work on the floor, in its evaluation.* | `tsstg_model.py` (`E205` classes `A8: sitting down`, `A6: pickup`, `A16/17: shoe kneeling`), `fall_monitor.py` (`compute_knee_angle`, `controlled_veto`, strict lying post-fall stability, `FR-6`), `evaluate_benchmark.py` (`FR-10`) | `T4` (sit-down & pickup discrimination on real weights), `T12` (Sub-cases a & b), `T16`, and held-out look-alike video evaluation | Real `MMAction2TSSTG` classifies sit-down (`A8`, `ntu60[7] > 0.80`) and pickup/bending (`A6`, `ntu60[5] > 0.50`) into `probs_7[5]` (`probs_7[6] < 0.10`, `T4`); seated `pred_idx == 6` false positives and rapid sit-downs/bends are vetoed (`0` false fall alerts in `T12a/b`); `lookalike_fa_breakdown` documents performance per activity. |
| **`AC-6`** | *The selected approach is evaluated on representative video sequences, with documented performance and limitations.* | `transfer_gate.py` (`FR-5`), `evaluate_benchmark.py` (`FR-10`) | `T9`, `T16`, and generated `evaluation_report.md` | Evaluates `FallMonitor` vs. `BaselineFallMonitor` on held-out video sequences + look-alike clips, reporting Precision, Recall, F1, FA/h, Detection Latency (s), and explicit surviving limitations (`L1`, `L2`, licensing). |

---

## 5. Phase-by-Phase Implementation Structure & Definition of DONE (`DoD`)

Implementation is divided into **5 sequential phases** (matching Tasks 1–5 of [`plan.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/brain/4f76fa57-e597-4563-a470-8a8d7031890c/plan.md)). No phase may be marked `DONE` until every checkbox in its **Definition of DONE** is verified.

---

### Phase 8.1 (Task 1): Core OpenMMLab MMAction2 / PYSKL 17-Keypoint ST-GCN Model (`tsstg_model.py`)
- **Objective**: Implement the standalone pure-PyTorch OpenMMLab `Graph(layout='coco', mode='stgcn_spatial')`, 10-stage `RecognizerGCN` (`STGCN` + `GCNHead`), multi-stream `TSSTGClassifier` (`MMAction2TSSTG`), and 60-to-7 class projection against the 8 verified checkpoints on `F:\fall_detection\weights`.
- **Requirements Covered**: `ENV-4`, `ENV-5`, `STORAGE-1`, `ASSET-2A`, `ASSET-2B`, `FR-5` (model core), `NFR-3`, `NFR-4`.
- **Files Created**:
  - `tsstg_model.py`
  - `tests/test_tsstg_model.py`
- **Definition of DONE (`DoD-8.1`)**:
  - [ ] All 8 official OpenMMLab MMAction2 v1.0 and PYSKL checkpoints reside on `F:\fall_detection\weights`, their SHA-256 hashes are pinned in `WEIGHT_SHA256_REGISTRY`, and they load via `torch.load(..., map_location=device, weights_only=True)`.
  - [ ] `Test T1` (`test_mmaction2_stgcn_ten_stages_matches_code_divergence`) passes: confirms 10 `STGCNBlock` stages, stage 0 `residual == 0`, stages `4` and `7` temporal stride `2`, channels `[64, 64, 64, 64, 128, 128, 128, 256, 256, 256]`, and `data_bn.num_features == 51`.
  - [ ] `Test T2` (`test_coco_17_graph_matches_checkpoint_buffer_before_load`) passes: confirms `normalize_digraph` computes asymmetric $\mathbf{A}\mathbf{D}^{-1}$ and `torch.allclose(model.backbone.gcn[0].gcn.A, sd['backbone.gcn.0.gcn.A'], atol=1e-6)` **before** calling `load_state_dict` across both MMAction2 v1.0 and PYSKL checkpoints.
  - [ ] `Test T3` (`test_mmaction2_and_pyskl_checkpoints_load_strictly_with_sha256`) passes: confirms `verify_sha256` and `load_recognizer_gcn_checkpoint(..., strict=True)` succeed with zero missing or unexpected keys.
  - [ ] `Test T4` (`test_mmaction2_tsstg_discriminates_fall_sitdown_and_pickup_on_real_weights`) passes: confirms that real pretrained `TSSTGClassifier` predicts `pred_idx_7 == 6` (`probs_7[6] > 0.90`, `A43: falling`) on a 17-keypoint falling sequence and `pred_idx_7 == 5` (`probs_7[6] < 0.10`) on both sit-down (`A8`, `ntu60[7] > 0.80`) and pickup/bending (`A6`, `ntu60[5] > 0.50`) sequences.

---

### Phase 8.2 (Task 2): Native 17-Keypoint MMAction2 Adapter, 25 Hz Resampling & Imputation-Jump Masking (`keypoint_adapter.py`)
- **Objective**: Convert variable-FPS, noisy 17-keypoint `yolo26x-pose.pt` tracks into finite, 25.0 Hz resampled `(1, 2, 100, 17, 3)` MMAction2 stream tensors (`j`, `jm`, `b`, `bm`) following the exact `PreNormalize2D -> GenSkeFeat -> UniformSampleFrames(100) -> FormatGCNInput(M=2)` pipeline order with imputation-jump masking.
- **Requirements Covered**: `FR-3`, `FR-4`, `NFR-1`, `NFR-4`.
- **Files Created**:
  - `keypoint_adapter.py`
  - `tests/test_keypoint_adapter.py`
- **Definition of DONE (`DoD-8.2`)**:
  - [ ] `Test T5` (`test_prenormalize2d_and_gen_ske_feat_coco_17_modalities`) passes: verifies `PreNormalize2D` (`[-1, 1]` relative to `(h, w)`) and 17-pair `COCO_BONE_PAIRS` + pairwise mean confidence in `gen_ske_feat`.
  - [ ] `Test T6` (`test_finite_imputation_and_imputation_jump_masking_in_jm_and_bm`) passes: verifies causal forward-fill / frame-0 bbox-center fallback, `torch.isfinite(tensor).all()`, and `jm[:, k, v, :2] == 0.0` (plus connected `bm`) when an imputed joint snaps back to a real detection.
  - [ ] `Test T7` (`test_uniform_sample_100_order_and_format_gcn_m2_zero_pad`) passes: verifies that `GenSkeFeat` runs before `UniformSampleFrames(100)` (`np.mod`) and `FormatGCNInput(num_person=2, mode='zero')` produces shape `(1, 2, 100, 17, 3)` with `m=1` zero-padded.
  - [ ] `Test T8` (`test_fps_resampling_to_25hz_and_gap_reset_policy`) passes: verifies that `12.5 Hz` and `25.0 Hz` samplings of the same physical motion produce identical stream tensors (`atol=1e-4`), and that any tracking gap `> hold_s` (`1.0s`) returns `None`.
  - [ ] Full test suite runs green (`T1`–`T8`).

---

### Phase 8.3 (Task 3): Pre-State-Machine `YOLO26 -> MMAction2TSSTG` Transfer & Calibration Gate (`transfer_gate.py`)
- **Objective**: Directly test `[UNTESTED DEPARTURE 1]` (transfer of NTU-60 HRNet 2D trained `MMAction2TSSTG` to `yolo26x-pose.pt` keypoints across `prenormalize2d` vs. `track_compact` and stream combinations) before building the downstream state machine.
- **Requirements Covered**: `FR-5`, `AC-6`.
- **Files Created**:
  - `transfer_gate.py`
  - `tests/test_transfer_gate.py`
- **Definition of DONE (`DoD-8.3`)**:
  - [ ] `Test T9` (`test_transfer_gate_evaluates_norm_modes_and_finetune_step`) passes: verifies `evaluate_tsstg_transfer_on_sequences` across `('prenormalize2d', 'track_compact')` on real `MMAction2TSSTG` weights and verifies that `finetune_tsstg_on_yolo26` reduces `CrossEntropyLoss` over 3 gradient steps while keeping graph buffer `A` invariant.
  - [ ] Full test suite runs green (`T1`–`T9`).

---

### Phase 8.4 (Task 4): Hybrid State Machine (`fall`, `lying`, `immobility`), Track Stitching, Windowed Excursion & Event Evidence (`fall_monitor.py`)
- **Objective**: Implement the upgraded `FallMonitor` (preserving `BaselineFallMonitor` and `S0`'s CLI/batch interface) with track stitching, `0.4s`-smoothed `MMAction2TSSTG` + `knee_angle` controlled-action veto, strict lying post-fall stability, 2.0s windowed confident-joint excursion for immobility, operator review lifecycle, and structured event evidence.
- **Requirements Covered**: `FR-1`, `FR-2`, `FR-6`, `FR-7`, `FR-8`, `FR-9`, `NFR-2`, `AC-1`, `AC-2`, `AC-3`, `AC-4`, `AC-5`.
- **Files Created**:
  - `fall_monitor.py`
  - `tests/test_fall_monitor.py`
- **Definition of DONE (`DoD-8.4`)**:
  - [ ] `Test T10` (`test_ref_scale_outlier_clamped_and_non_decaying`) passes.
  - [ ] `Test T11` (`test_track_stitching_and_cold_start_under_30_frames`) passes.
  - [ ] `Test T12` (`test_hybrid_fall_latch_requires_descent_and_lying_stability_and_vetoes_seated`) passes.
  - [ ] `Test T13` (`test_lying_candidate_hold_runs_and_merge_runs_semantics`) passes.
  - [ ] `Test T14` (`test_windowed_displacement_immobility_ignores_jitter_and_imputed_joints`) passes.
  - [ ] `Test T15` (`test_candidate_lifecycle_dedup_event_evidence_and_cli_run`) passes.
  - [ ] Full test suite runs green (`T1`–`T15`).

---

### Phase 8.5 (Task 5): Held-Out Video & Look-Alike Evaluation (`evaluate_benchmark.py`)
- **Objective**: Evaluate `FallMonitor` vs. `BaselineFallMonitor` on held-out video sequences (Le2i + look-alike activities `bending`, `kneeling`, `sitting`, `floor_work`) with ground-truth annotations and produce the final documented performance and limitations report.
- **Requirements Covered**: `FR-10`, `ASSET-3`, `LIC-1`, `LIC-2`, `LIC-3`, `AC-5`, `AC-6`.
- **Files Created**:
  - `evaluate_benchmark.py`
  - `tests/test_evaluate_benchmark.py`
  - `evaluation_report.md` (generated output report)
- **Definition of DONE (`DoD-8.5` & Project-Wide Final DoD)**:
  - [ ] `Test T16` (`test_le2i_annotation_parser_and_heldout_benchmark_metrics`) passes.
  - [ ] All 16 automated tests (`T1`–`T16`) pass in a single `F:\fall_detection\.venv\Scripts\python.exe -m pytest` run with 0 failures and 0 errors.
  - [ ] `evaluate_benchmark.py` produces `evaluation_report.md` comparing `BaselineFallMonitor` (`S0`) vs. upgraded `FallMonitor` across Precision, Recall, F1, FA/h, Mean Detection Latency (`s`), Per-Activity Look-Alike False Alarms (`bending`, `kneeling`, `sitting`, `floor_work`), and documented limitations.
