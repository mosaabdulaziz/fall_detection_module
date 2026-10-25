# OpenMMLab MMAction2 / PYSKL 17-COCO-Keypoint Multi-Stream ST-GCN (`MMAction2TSSTG`) + Transfer Gate, Hybrid Veto, Windowed-Excursion Immobility & Event-Evidence Fall Monitor Implementation Plan

> **For agentic workers:** implement task by task. Steps use `- [ ]` checkboxes.
> Test first: write the failing test, watch it fail for the expected reason,
> write the minimal implementation, watch it pass, then run the full suite.

**Goal:** Build and evaluate a modular, operator-reviewable human fall, persistent lying, and prolonged unusual immobility candidate detection pipeline on top of `yolo26x-pose.pt` (`ultralytics >= 8.4.0`) + `ByteTrack` and **OpenMMLab MMAction2 v1.0 / PYSKL 17-COCO-keypoint Multi-Stream ST-GCN** (`RecognizerGCN`, replacing `yysijie/st-gcn` / `GajuuzZ`), verified via a held-out transfer gate and ground-truth video evaluation covering bending (`pickup`), kneeling (`wear/take off a shoe`), sitting (`sitting down`), and floor work.

**Architecture:** The pipeline is factored into five modules under `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\` (with all heavy model checkpoints stored on `F:\fall_detection\weights`):
1. [`tsstg_model.py`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/tsstg_model.py): Vendors the standalone pure-PyTorch OpenMMLab MMAction2 `Graph(layout='coco', mode='stgcn_spatial')` (`num_node=17`, asymmetric `normalize_digraph` $\mathbf{A}\mathbf{D}^{-1}$, and `(3, 17, 17)` spatial partition verified against `state_dict['backbone.gcn.0.gcn.A']` **before** loading), 10-stage `STGCN` backbone + `GCNHead` (`RecognizerGCN`), and multi-stream `TSSTGClassifier` (`MMAction2TSSTG` / `TwoStreamSpatialTemporalGraph` drop-in replacement) loading official Apache-2.0 checkpoints from `F:\fall_detection\weights` with `weights_only=True`, pinned SHA-256 verification, and seamless 60-to-7 class projection (`probs_7` + `ntu60_probs`).
2. [`keypoint_adapter.py`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/keypoint_adapter.py): Implements native 17-COCO-keypoint sanitization (`is_confident` mask + causal forward-fill imputation), FPS-aware temporal resampling onto a uniform `25.0 Hz` (`dt = 0.04s`, `1.16s`) 30-frame grid, track-gap `> hold_s` (`1.0s`) buffer reset, and OpenMMLab MMAction2's exact preprocessing pipeline order: `PreNormalize2D` $\rightarrow$ `GenSkeFeat(dataset='coco', feats=['j', 'jm', 'b', 'bm'])` with **imputation-jump masking** (`jm = 0`, `bm = 0` on imputed transitions) $\rightarrow$ `UniformSampleFrames(clip_len=100, test_mode=True)` (`np.mod(np.arange(100), T)`) $\rightarrow$ `FormatGCNInput(num_person=2, mode='zero')` (`(1, 2, 100, 17, 3)`).
3. [`transfer_gate.py`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/transfer_gate.py): Runs a pre-state-machine transfer experiment feeding `yolo26x-pose.pt` 17-COCO-keypoint skeletons from held-out fall, sit-down (`A8`), pickup/bending (`A6`), and shoe-kneeling (`A16/A17`) sequences through `TSSTGClassifier`, compares MMAction2 v1.0 vs. PYSKL checkpoints, stream combinations (`j+jm`, `j+b`, `2*j+2*b+jm+bm`), and normalization modes (`prenormalize2d` vs. `track_compact`), and provides `finetune_tsstg_on_yolo26()` if held-out macro-F1 `< 0.80`.
4. [`fall_monitor.py`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/fall_monitor.py): Preserves the baseline [`fall_monitor.py`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py) (`S0`) batch interface `FallMonitor.update(t, ids, boxes, kps)` and CLI `run(a)`, retains `BaselineFallMonitor` (`S0`) for side-by-side benchmarking, and upgrades `FallMonitor` with spatial-temporal track stitching (`stitch_s = 1.5s`), `0.4s`-smoothed `MMAction2TSSTG` + lower-limb (`knee_angle`) controlled-action veto, a two-phase Pending-Fall Latch with strict lying post-fall stability, 2.0-second windowed confident-joint excursion (`disp_win`) for prolonged immobility, operator lifecycle management (`candidate` $\rightarrow$ `acknowledged` / `confirmed` / `dismissed`), and structured event evidence (`snapshot_path`, `clip_window`, `tsstg_probs`, `ntu60_top5`, `kinematics`).
5. [`evaluate_benchmark.py`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/evaluate_benchmark.py): Evaluates `FallMonitor` vs. `BaselineFallMonitor` on held-out Le2i (`Office`, `Lecture room`, or full Le2i since MMAction2/PYSKL weights were trained on NTU-60, zero Le2i leakage!) and look-alike/ADL clips (bending, kneeling, sitting, ordinary floor work, and URFD/MCFD clips), reporting **Precision, Recall, F1, False Alarms per Hour (FA/h), Detection Latency (s), and Per-Activity False-Alarm Breakdown**.

**Tech stack:** Python 3.10+ (`F:\fall_detection\.venv`), PyTorch (`torch >= 2.0`), NumPy (`numpy >= 1.24`), Pandas (`pandas >= 2.0`), OpenCV (`opencv-python >= 4.8`), Ultralytics (`ultralytics >= 8.4.0` — required for `yolo26x-pose.pt` and `bytetrack.yaml`), PyYAML (`pyyaml >= 6.0`), `pytest` (no `mmcv`/`mmengine` runtime dependency required).

**Design:** [`research/design/design-3.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/design/design-3.md) (the plan argues from the design; the design travels with it)

**Evidence:** [`research/ledger/evidence.jsonl`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/ledger/evidence.jsonl) — every `[SOURCE]` line below resolves to an entry there (`E01`–`E05`, `E101`–`E111b`, `E201`–`E205`).

---

## Global constraints, Storage & Licensing (`FO2`, `E201`)
- **Storage Rule**: Never download model weights or datasets onto the `C:` drive; store all checkpoints under `F:\fall_detection\weights\`.
- Pin `ultralytics >= 8.4.0` (the release where `yolo26x-pose.pt` first shipped) with `tracker="bytetrack.yaml"` (`persist=True`, `conf_det=0.25`) (`problem.md` §3; `E01`; `FO2`).
- Verify all OpenMMLab MMAction2 v1.0 / PYSKL `.pth` checkpoints on `F:\fall_detection\weights\` against their pinned SHA-256 digests (`WEIGHT_SHA256_REGISTRY`) and load exclusively with `torch.load(weight_path, map_location=device, weights_only=True)` (`E201`, `FO2`):
  - `mmaction2_joint.pth` (`12,479,895` B): `484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea`
  - `mmaction2_bone.pth` (`12,479,895` B): `c4b44488123d68e19d742ade26d02ff638a911bb0084fcfa7b54d751957318dd`
  - `mmaction2_jm.pth` (`12,479,895` B): `f18eb408a42eebced02fc0e0dfeed2c6edb320cc1752abc6df855dff0845fc6d`
  - `mmaction2_bm.pth` (`12,479,895` B): `99c60e2d462a3978114193f35b5fce465bfa32de2275325913ab8a394900bfb0`
  - `pyskl_j.pth` (`12,474,519` B): `b4e476a34f2fe641f111cf8b8b66d08fca163a0b45e3df6b9ebb5c93ac9a71b2`
  - `pyskl_b.pth` (`12,474,519` B): `e9d7cae55964837e6bd9ba5b00f346655da1937317421d79a8963fa9cf0c7fed`
  - `pyskl_jm.pth` (`12,474,519` B): `131738564797c3d1c4fc5f3580e4cda0303cffa16efb5f46f4c924e6d9488615`
  - `pyskl_bm.pth` (`12,474,519` B): `0e55b74c7d50410ec64ddb4527be6c83145ced62ed671383df45ccd1a2225ac4`
- Resample per-track 17-keypoint skeleton windows onto a uniform `25.0 Hz` (`dt = 0.04s`, `30` frames = `1.16s`) grid, reset the buffer on any tracking gap `> hold_s` (`1.0s`), compute `GenSkeFeat` (`j`, `jm`, `b`, `bm`) with **imputation-jump masking** **before** `UniformSampleFrames(clip_len=100, test_mode=True)` (`np.mod(np.arange(100), T)`), and format with `FormatGCNInput(num_person=2, mode='zero')` (`E204`, `FO7`).
- **Licensing Notice (`FO2`, `E201`)**: OpenMMLab `MMAction2` and `PYSKL` code and model weights are licensed under **Apache-2.0**; `ultralytics` is licensed under **AGPL-3.0**; Le2i is `CC-BY-NC-SA-3.0` and NTU-RGB+D 60 has academic research dataset terms.

---

## Decisions carried from the design
| Decision | Value | Source | Locator |
|---|---|---|---|
| Stage 1 detector & tracker | `yolo26x-pose.pt` (`ultralytics >= 8.4.0`) + `bytetrack.yaml` (`conf_det=0.25`) | `E01`, `FO2` | `S0`, `fall_monitor.py` L199–200, L236–240 |
| Streaming batch interface & downward velocity sign | `FallMonitor.update(t, ids, boxes, kps)`; `vdown = -np.diff(hip_rel) / dt > 0.15` (downward = positive) | `E05` | `S0`, `fall_monitor.py` L68, L125, L151 |
| Two-level alert deduplication | `cooldown_s = 5.0` (`ev['t_candidate'] <= last_cand + cooldown_s`); `realert_s = 60.0` (`t - emitted < realert_s`) | `E04`, `E05` | `S0`, `fall_monitor.py` L33, L93–98, L167–168 |
| Native 17-COCO-keypoint graph & asymmetric `normalize_digraph` | `Graph(layout='coco', mode='stgcn_spatial')`: `num_node=17`, `center=0`, 16 `inward` edges, `normalize_digraph` ($\mathbf{A}\mathbf{D}^{-1}$ with `Dl[i]**(-1)`), `(3, 17, 17)` spatial partition `[a_close(hop=0), a_close(hop=1), a_further(hop=1)]` | `E202`, `E111b`, `FO1` | `S2`, `open-mmlab/mmaction2`, `mmaction/models/utils/graph.py` L57–78, L167–205 |
| Standalone `RecognizerGCN` (`STGCN` + `GCNHead`) & multi-stream fusion | 10 `STGCNBlock` stages (`BatchNorm1d(51)` `VC`, `inflate_stages=[5,8]`, `down_stages=[5,8]`, `A * PA`), `GCNHead` (`AdaptiveAvgPool2d(1)`, `mean(dim=1)`, `Linear(256, 60)`), multi-stream probability fusion (`j`, `jm`, `b`, `bm`) | `E201`, `E203` | `S2`, `mmaction/models/backbones/stgcn.py` L153–238, `gcn_head.py` L25–71 |
| Exact MMAction2 preprocessing order + FPS resampling + imputation-jump masking | Resample `[t - 1.16s, t]` to 30 frames (`25.0 Hz`); `PreNormalize2D` $\rightarrow$ `GenSkeFeat` (`j`, `jm`, `b`, `bm` with `jm/bm = 0` on imputed transitions) $\rightarrow$ `UniformSampleFrames(100)` (`np.mod`) $\rightarrow$ `FormatGCNInput(num_person=2, mode='zero')` | `E204`, `[DECISION]` | `S2`, `mmaction/datasets/transforms/pose_transforms.py` L710–1281, `formatting.py` L388–445; `design-3.md` §2 Stage 2 (`FO7`) |
| NTU-60 to 7-class canonical projection | `42 ('falling') -> class 6 ('Fall Down')`; `[5 ('pickup'), 7 ('sitting down'), 15 ('wear a shoe'), 16 ('take off a shoe')] -> class 5 ('Sit down')`; `8 ('standing up') -> class 4 ('Stand up')`; `[25,26,41,58,59] -> class 1 ('Walking')`; `[10,11,28,29] -> class 2 ('Sitting')` | `E205`, `[DECISION]` | `S2`, `tools/data/skeleton/label_map_ntu60.txt`; `design-3.md` §2 Stage 3 |
| Pre-state-machine Transfer Gate | Evaluate `yolo26x-pose.pt` 17-keypoint skeletons across MMAction2 vs. PYSKL weights, stream combinations (`j+jm`, `j+b`, `4s`), and `prenormalize2d` vs. `track_compact`; provide `finetune_tsstg_on_yolo26()` | `[DECISION]` | `design-3.md` §2 Stage 3 (`[UNTESTED DEPARTURE 1]`, resolves `FO3`) |
| Smoothed `MMAction2TSSTG` + lower-limb `knee_angle` controlled-action veto | Smooth `probs_7` over `0.4s` (`min_action_score = 0.50`); veto both Path A & Path B when `probs_smooth[[0,1,2,4,5]].max() > max(probs_smooth[3], probs_smooth[6])` or controlled bend/kneel (`angle < 45`, `aspect < 0.85`, `head_rel > 0.30`, `vpk <= 0.30`); require lying posture for post-fall stability | `[DECISION]` | `design-3.md` §2 Stage 4 (`[UNTESTED DEPARTURE 3]`, resolves `O6`, `PO2`, `PO3`, `FO4`) |
| Windowed confident-joint excursion (`disp_win`) & outlier-clamped `ref_scale` | `disp_win <= 0.035` over `2.0s` on confident joints (`conf >= 0.3`, `>= 3` frames); `ref_scale` = 90th-pct of 60s `hypot(w, h)` clamped to `<= 1.15 * median` and floored by `ref_scale_init`; `T_immobile_post_fall=40.0s`, `T_immobile_lying=100.0s`, `T_immobile_seated=150.0s` | `[DECISION]` | `design-3.md` §2 Stage 4 (`[UNTESTED DEPARTURE 4]`, resolves `O3`, `O9`, `FO5`) |
| Spatial-temporal track stitching & `< 30` frame fallback | Stitch new `tid` within `stitch_s = 1.5s` and `0.8 * ref_scale` of lost track; run `S0` kinematic rules when track span `< 1.16s` (`< 6` frames) | `[DECISION]` | `design-3.md` §2 Stage 1 & 4 (`[UNTESTED DEPARTURE 5]`, resolves `O5`, `FO6`) |

---

## Known paper/code divergences to guard
| Divergence | Paper says | Code does | Guard test |
|---|---|---|---|
| ST-GCN layer count per stream (`E111a`, `E203`) | 9 layers of ST-GCN operators (`S1`, `arXiv:1801.07455` §3.3) | 10 `STGCNBlock` stages (`num_stages=10`: Stage 0 `3->64` with `residual=False`, Stages 1–3 `64`, Stage 4 `128 stride=2`, Stages 5–6 `128`, Stage 7 `256 stride=2`, Stages 8–9 `256`) in `mmaction/models/backbones/stgcn.py` L160–221 | `T1` |
| Digraph adjacency normalization & spatial partition (`E111b`, `E202`, `FO1`) | Symmetric $\Lambda^{-1/2}\mathbf{A}\Lambda^{-1/2}$ (`S1`, `arXiv:1801.07455` §3.3 Eq. 3) | `mmaction/models/utils/graph.py` L57–78, L183–205 calls `normalize_digraph` ($\mathbf{A}\mathbf{D}^{-1}$ with `Dl[i]**(-1)`) and constructs `(3, 17, 17)` `stgcn_spatial` with `a_close` when `hop_dis[j, center] >= hop_dis[i, center]`, verified against `state_dict['backbone.gcn.0.gcn.A']` | `T2` |
| Head layer key naming (`E203`) | Unified `GCNHead` | MMAction2 v1.0 checkpoints use `cls_head.fc.{weight,bias}` while PYSKL checkpoints use `cls_head.fc_cls.{weight,bias}` | `T3` |

---

## Tasks

### Task 1: `tsstg_model.py` — Standalone OpenMMLab MMAction2 `Graph(layout='coco', mode='stgcn_spatial')`, 10-Stage `RecognizerGCN`, Pre-Load `A` Buffer Check, Multi-Stream Fusion & 60-to-7 Class Projection (`E201`–`E205`, `FO1`, `FO2`, `FO8`)

**Files:**
- Create: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\tsstg_model.py`
- Weights (already downloaded & verified on `F:` drive): `F:\fall_detection\weights\{mmaction2_joint.pth, mmaction2_bone.pth, mmaction2_jm.pth, mmaction2_bm.pth, pyskl_j.pth, pyskl_b.pth, pyskl_jm.pth, pyskl_bm.pth}`
- Test: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\tests\test_tsstg_model.py`

**Interfaces & Exact Upstream Implementation (`mmaction/models/utils/graph.py`, `gcn_utils.py`, `stgcn.py`, `gcn_head.py`):**
- Consumes: Dictionary of per-stream formatted tensors `stream_tensors: dict[str, torch.Tensor]` where each tensor has shape `(N, M=2, T=100, V=17, C=3)` (or tuple `(j_tensor, jm_tensor)` for drop-in 2-stream calls) (`E203`, `E204`).
- Produces:
  1. `CLASS_NAMES: tuple[str, ...] = ('Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down')` and `NTU60_CLASS_NAMES: tuple[str, ...]` (60 classes from `label_map_ntu60.txt`, where index `42` is `'falling'`, `7` is `'sitting down'`, `8` is `'standing up (from sitting position)'`, `5` is `'pickup'`, `15` is `'wear a shoe'`, `16` is `'take off a shoe'`, `41` is `'staggering'`) (`E205`).
  2. `WEIGHT_SHA256_REGISTRY: dict[str, str]` mapping all 8 official checkpoint filenames on `F:\fall_detection\weights` to their verified SHA-256 digests (`E201`, `FO2`).
  3. `normalize_digraph(A: np.ndarray, dim: int = 0) -> np.ndarray` and `Graph(layout: str = 'coco', mode: str = 'stgcn_spatial', max_hop: int = 1)` (`mmaction/models/utils/graph.py` L57–205, `E202`).
  4. `unit_gcn`, `unit_tcn`, `STGCNBlock`, `STGCN`, `GCNHead`, `RecognizerGCN` (`mmaction/models/utils/gcn_utils.py`, `mmaction/models/backbones/stgcn.py`, `mmaction/models/heads/gcn_head.py`, `E203`).
  5. `verify_sha256(path: str | Path, expected_sha256: str | None = None) -> str` and `load_recognizer_gcn_checkpoint(model: RecognizerGCN, weight_path: str | Path, device: str = 'cpu', expected_sha256: str | None = None) -> RecognizerGCN`:
     - Verifies SHA-256, loads via `torch.load(weight_path, map_location=device, weights_only=True)`, extracts `sd = dict(ckpt.get('state_dict', ckpt))`, remaps `cls_head.fc_cls.{weight,bias} -> cls_head.fc.{weight,bias}` if present, asserts `torch.allclose(model.backbone.gcn[0].gcn.A, sd['backbone.gcn.0.gcn.A'], atol=1e-6)` **before** `model.load_state_dict(sd, strict=True)` (`FO1`, `T2`), and sets `.eval()`.
  6. `project_ntu60_to_7_classes(ntu60_probs: np.ndarray | torch.Tensor) -> np.ndarray` and `TSSTGClassifier(weight_dir: str | Path = 'F:/fall_detection/weights', variant: str = 'mmaction2', streams: Sequence[str] = ('j', 'jm'), stream_weights: Sequence[float] | None = None, device: str = 'cpu')`:
     - Loads one `RecognizerGCN` per requested stream (`'j'`, `'jm'`, `'b'`, `'bm'`), computes stream-weighted softmax probability fusion `ntu60_probs` `(60,)`, projects to `probs_7` `(7,)`, and returns `(pred_idx_7, class_name_7, probs_7, ntu60_probs)`.

- [ ] **Step 1: Write the failing tests (`T1`, `T2`, `T3`, `T4`) in `tests/test_tsstg_model.py`**
- [ ] **Step 2: Run them and confirm they fail for the right reason**

Run: `F:\fall_detection\.venv\Scripts\python.exe -m pytest tests/test_tsstg_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tsstg_model'`.

- [ ] **Step 3: Implement `tsstg_model.py` against the verified checkpoints in `F:\fall_detection\weights`**
- [ ] **Step 4: Run `pytest tests/test_tsstg_model.py` and confirm `T1`–`T4` pass**
- [ ] **Step 5: Run the full project suite**
- [ ] **Step 6: Commit**

---

### Task 2: `keypoint_adapter.py` — Native 17-COCO-Keypoint Sanitizer, FPS-Aware 25 Hz Resampling, Gap Reset, Imputation-Jump Masking, and Exact MMAction2 `PreNormalize2D -> GenSkeFeat -> UniformSampleFrames(100) -> FormatGCNInput(M=2)` Pipeline (`E204`, `FO7`)

**Files:**
- Create: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\keypoint_adapter.py`
- Test: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\tests\test_keypoint_adapter.py`

**Interfaces:**
- Consumes: Timestamped per-track history of 17-keypoint arrays `kp_17: np.ndarray` `(17, 3)`, bounding boxes `box_xyxy: np.ndarray` `(4,)`, and `img_shape: tuple[int, int] = (1080, 1920)`.
- Produces:
  1. `COCO_BONE_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))` (`mmaction/datasets/transforms/pose_transforms.py` L791–793, `E204`).
  2. `sanitize_keypoints_17(kp_17: np.ndarray, box_xyxy: np.ndarray, prev_kp_17: np.ndarray | None, conf_kp: float = 0.3, score_mode: str = 'raw') -> tuple[np.ndarray, np.ndarray]`:
     - Returns `(clean_kp_17, is_confident_17)` where `clean_kp_17` has shape `(17, 3)` (`np.isfinite(clean_kp_17).all() == True`) and `is_confident_17` is a boolean array of shape `(17,)` (`True` only where the input joint had finite coordinates, `(x, y) != (0, 0)`, and `conf >= conf_kp`). Non-confident joints are imputed via causal forward-fill from `prev_kp_17` (or bbox center `((x1+x2)/2, (y1+y2)/2)` at frame 0) with `score = conf_kp`.
  3. `resample_window_25hz(timestamps: Sequence[float], kp17_list: Sequence[np.ndarray], conf_mask_list: Sequence[np.ndarray], window_s: float = 1.16, num_frames: int = 30, max_gap_s: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None`:
     - Returns `None` if `len(timestamps) < 6`, if `timestamps[-1] - timestamps[0] < window_s - 1e-3`, or if any consecutive step inside `[t_end - window_s, t_end]` has `t[k] - t[k-1] > max_gap_s` (`1.0s` gap reset policy, `FO7`).
     - Otherwise interpolates `(x, y, score)` onto `t_grid = np.linspace(t_end - window_s, t_end, num_frames)` (`25.0 Hz`, `dt = 0.04s`, shape `(30, 17, 3)`) and propagates `conf_mask_30` `(30, 17)`.
  4. `prenormalize_2d(kp_t_17_3: np.ndarray, img_shape: tuple[int, int] = (1080, 1920), norm_mode: str = 'prenormalize2d') -> np.ndarray`:
     - Exact MMAction2 `PreNormalize2D` (`(x - w/2)/(w/2)`, `(y - h/2)/(h/2)` for `norm_mode='prenormalize2d'`, or subject-centered padded crop for `norm_mode='track_compact'`) (`E204`).
  5. `gen_ske_feat(j_norm: np.ndarray, conf_mask: np.ndarray | None = None, streams: Sequence[str] = ('j', 'jm', 'b', 'bm')) -> dict[str, np.ndarray]`:
     - Computes requested streams (`'j'`, `'jm'`, `'b'`, `'bm'`) of shape `(1, T, 17, 3)` **before** `UniformSampleFrames`, and applies **imputation-jump masking** (`jm[:, k, v, :2] = 0.0` when `not (conf_mask[k, v] and conf_mask[k+1, v])`, and `bm[:, k, v1, :2] = 0.0` when either bone endpoint `(v1, v2)` is non-confident at `k` or `k+1`) (`E204`, `FO7`).
  6. `uniform_sample_and_format_gcn(feat_dict: dict[str, np.ndarray], clip_len: int = 100, num_person: int = 2, device: str = 'cpu') -> dict[str, torch.Tensor]`:
     - Samples time dimension `T -> clip_len` (`100`) using MMAction2's exact `UniformSampleFrames(test_mode=True)` index logic (`inds = np.mod(np.arange(clip_len), T)` when `T < clip_len`) and zero-pads `M=1 -> num_person=2` via `FormatGCNInput(num_person=2, mode='zero')`, returning tensors of shape `(1, 2, 100, 17, 3)` (`E204`).
  7. `build_mmaction2_tensors(pts_30_17: np.ndarray, conf_mask_30_17: np.ndarray | None = None, img_shape: tuple[int, int] = (1080, 1920), streams: Sequence[str] = ('j', 'jm'), norm_mode: str = 'prenormalize2d', clip_len: int = 100, device: str = 'cpu') -> dict[str, torch.Tensor]`: end-to-end helper chaining `prenormalize_2d -> gen_ske_feat -> uniform_sample_and_format_gcn`.

- [ ] **Step 1: Write the failing tests (`T5`, `T6`, `T7`, `T8`) in `tests/test_keypoint_adapter.py`**
- [ ] **Step 2: Run them and confirm they fail for the right reason**
- [ ] **Step 3: Implement `keypoint_adapter.py`**
- [ ] **Step 4: Run `pytest tests/test_keypoint_adapter.py` and confirm `T5`–`T8` pass**
- [ ] **Step 5: Run the full project suite**
- [ ] **Step 6: Commit**

---

### Task 3: `transfer_gate.py` — Pre-State-Machine `YOLO26 -> MMAction2TSSTG` Transfer Experiment, Stream/Variant Selection & Fine-Tuning Hook (`FO3`, `E201`–`E205`)

**Files:**
- Create: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\transfer_gate.py`
- Test: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\tests\test_transfer_gate.py`

**Interfaces:**
- Consumes: `TSSTGClassifier` (`tsstg_model.py`) and `build_mmaction2_tensors` (`keypoint_adapter.py`).
- Produces:
  1. `evaluate_tsstg_transfer_on_sequences(sequences: list[dict], classifier: TSSTGClassifier, norm_modes: Sequence[str] = ('prenormalize2d', 'track_compact')) -> dict[str, Any]`:
     - Evaluates `TSSTGClassifier` on labeled 17-keypoint sequence windows (`Fall Down`, `Sit down`, `Standing`, `Walking`), returning per-mode `fall_recall`, `sitting_specificity`, `pickup_specificity`, `macro_f1`, and `recommended_norm_mode`.
  2. `finetune_tsstg_on_yolo26(train_windows: np.ndarray, train_labels_ntu60: np.ndarray, base_weight_path: str | Path, out_weight_path: str | Path, epochs: int = 5, lr: float = 1e-4, device: str = 'cpu') -> dict[str, float]`:
     - Fine-tunes `RecognizerGCN` (`CrossEntropyLoss` on 60-class logits) on `yolo26x-pose.pt` extracted 17-keypoint windows while preserving buffer `A` (`FO3`).

- [ ] **Step 1: Write the failing test (`T9`) in `tests/test_transfer_gate.py`**
- [ ] **Step 2: Run it and confirm it fails for the right reason**
- [ ] **Step 3: Implement `transfer_gate.py`**
- [ ] **Step 4: Run `pytest tests/test_transfer_gate.py` and confirm `T9` passes**
- [ ] **Step 5: Run the full project suite**
- [ ] **Step 6: Commit**

---

### Task 4: `fall_monitor.py` — Hybrid State Machine (`fall`, `lying`, `immobility`), Track Stitching, Windowed Excursion, Operator Lifecycle & Event Evidence (`FO4`–`FO6`, `FO9`, `FO11`)

**Files:**
- Create: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\fall_monitor.py`
- Test: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\tests\test_fall_monitor.py`

**Interfaces:**
- Consumes: `TSSTGClassifier` (`tsstg_model.py`), `sanitize_keypoints_17`, `resample_window_25hz`, `build_mmaction2_tensors` (`keypoint_adapter.py`).
- Produces:
  1. `THRESHOLDS: dict` (same calibrated thresholds as `design-3.md` §2 Stage 4, with `weight_dir='F:/fall_detection/weights'`, `variant='mmaction2'`, `streams=('j', 'jm')`, `norm_mode='prenormalize2d'`).
  2. `BaselineFallMonitor`: exact `S0` implementation (`fall_monitor.py` L145–170) preserved for side-by-side comparison (`FO11`).
  3. `update_ref_scale(diag_history: list[float], box_xyxy: np.ndarray, max_growth: float = 1.15) -> float` (`FO5`).
  4. `compute_windowed_displacement(t_hist: np.ndarray, kp17_hist: np.ndarray, conf_mask_hist: np.ndarray, ref_scale: float, win_s: float = 2.0, min_valid_frames: int = 3) -> float` (`FO5`).
  5. `compute_knee_angle(kp_17: np.ndarray, conf_kp: float = 0.3) -> float` (`E01`, `FO4`).
  6. `FallMonitor(cfg=CFG, buffer_s=30.0, eval_every_s=0.0, realert_s=60.0, weight_dir='F:/fall_detection/weights', variant='mmaction2', streams=('j', 'jm'), device='cpu', classifier=None, evidence_dir=None, img_shape=(1080, 1920))`:
     - `.update(t: float, ids: Sequence[int], boxes: Sequence[np.ndarray], kps: Sequence[np.ndarray], frame: np.ndarray | None = None) -> list[dict]` (`E05`, `FO9`).
     - `.review_event(event_id: str, status: str, operator_note: str = '') -> dict` (`FO9`).
  7. `run(a)` and `main()` CLI (`FO9`).

- [ ] **Step 1: Write the failing tests (`T10`, `T11`, `T12`, `T13`, `T14`, `T15`) in `tests/test_fall_monitor.py`**
- [ ] **Step 2: Run them and confirm they fail for the right reason**
- [ ] **Step 3: Implement `fall_monitor.py`**
- [ ] **Step 4: Run `pytest tests/test_fall_monitor.py` and confirm `T10`–`T15` pass**
- [ ] **Step 5: Run the full project suite**
- [ ] **Step 6: Commit**

---

### Task 5: `evaluate_benchmark.py` & `tests/test_evaluate_benchmark.py` — Held-Out Le2i & Look-Alike Video Evaluation with Ground-Truth Metrics (`FO10`)

**Files:**
- Create: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\evaluate_benchmark.py`
- Test: `C:\Users\Maha Mamdouh\.gemini\antigravity\scratch\fall_detection_research\tests\test_evaluate_benchmark.py`

**Interfaces:**
- Consumes: `FallMonitor`, `BaselineFallMonitor` (`fall_monitor.py`).
- Produces:
  1. `parse_le2i_annotation(txt_path: str | Path, fps: float = 25.0) -> dict[str, Any]` (`FO10`).
  2. `compute_benchmark_metrics(predictions: list[dict[str, Any]], total_duration_hours: float) -> dict[str, Any]` (`FO10`).
  3. `evaluate_video_manifest(manifest_path: str | Path, out_report_md: str | Path, pose_weights: str = 'yolo26x-pose.pt', weight_dir: str = 'F:/fall_detection/weights') -> dict[str, Any]` (`FO10`).

- [ ] **Step 1: Write the failing tests (`T16`) in `tests/test_evaluate_benchmark.py`**
- [ ] **Step 2: Run them and confirm they fail for the right reason**
- [ ] **Step 3: Implement `evaluate_benchmark.py`**
- [ ] **Step 4: Run `pytest tests/test_evaluate_benchmark.py` and confirm `T16` passes**
- [ ] **Step 5: Run the full project suite**
- [ ] **Step 6: Commit**

---

## Test suite with source mapping

| ID | Test | Verifies | Pass criterion | Source mapping |
|---|---|---|---|---|
| `T1` | `test_mmaction2_stgcn_ten_stages_matches_code_divergence` | Guard for Paper-vs-Code divergence `E111a` / `E203` (10 `STGCNBlock` stages vs. 9 in paper) | `len(model.backbone.gcn) == 10`, stage 0 `residual` returns `0`, stages `4` and `7` `tcn.stride == 2`, channels `[64, 64, 64, 64, 128, 128, 128, 256, 256, 256]`, `data_bn.num_features == 51` | `E111a`, `E203` (`S2` `open-mmlab/mmaction2`, `mmaction/models/backbones/stgcn.py` L153–238) |
| `T2` | `test_coco_17_graph_matches_checkpoint_buffer_before_load` | Guard for `E111b`, `E202`, and `FO1` (asymmetric `normalize_digraph` $\mathbf{A}\mathbf{D}^{-1}$ and `(3, 17, 17)` `stgcn_spatial` partition) | **Before** calling `load_state_dict`, `torch.allclose(model.backbone.gcn[0].gcn.A, sd['backbone.gcn.0.gcn.A'], atol=1e-6)` passes across both MMAction2 v1.0 and PYSKL checkpoints; `normalize_digraph` matches $\mathbf{A}\mathbf{D}^{-1}$ | `E111b`, `E202` (`S2` `mmaction/models/utils/graph.py` L57–205), `FO1` |
| `T3` | `test_mmaction2_and_pyskl_checkpoints_load_strictly_with_sha256` | Official OpenMMLab checkpoints on `F:\fall_detection\weights` pass SHA-256 verification and load with `weights_only=True` and `strict=True` (`E201`, `E203`, `FO2`) | `verify_sha256` matches pinned SHA-256 for `mmaction2_joint.pth`, `mmaction2_jm.pth`, `mmaction2_bone.pth`, `pyskl_j.pth`, `pyskl_jm.pth`; `load_recognizer_gcn_checkpoint` succeeds with `strict=True` (0 missing/unexpected keys) | `E201`, `E203` (`S2` `configs/skeleton/stgcn/README.md` & `configs/stgcn/README.md`), `FO2` |
| `T4` | `test_mmaction2_tsstg_discriminates_fall_sitdown_and_pickup_on_real_weights` | Real pretrained `MMAction2TSSTG` (`j+jm` / `j+b`) accurately classifies 17-keypoint falling (`A43`, idx `42`), sitting down (`A8`, idx `7`), and pickup/bending (`A6`, idx `5`) (`E204`, `E205`, `FO8`) | On deterministic 17-keypoint sequences formatted with `build_mmaction2_tensors`, real pretrained `TSSTGClassifier` predicts `pred_idx_7 == 6` (`Fall Down`, `probs_7[6] > 0.90`) on the fall clip, and `pred_idx_7 == 5` (`Sit down` / controlled bend, `probs_7[6] < 0.10`) on both the sit-down (`ntu60[7] > 0.80`) and pickup/bending (`ntu60[5] > 0.50`) clips | `E201`, `E204`, `E205` (`S2` `open-mmlab/mmaction2`), `FO8` |
| `T5` | `test_prenormalize2d_and_gen_ske_feat_coco_17_modalities` | Exact MMAction2 `PreNormalize2D` (`[-1, 1]` relative to `(h, w)`) and `GenSkeFeat(dataset='coco', feats=['j','jm','b','bm'])` (`E204`) | `prenormalize_2d` maps `(w/2, h/2) -> (0, 0)` and `(w, h) -> (1, 1)`; `gen_ske_feat` computes exact 17-pair `COCO_BONE_PAIRS` differences and pairwise mean confidence `0.5 * (c_1 + c_2)` for `b`, `jm`, `bm` | `E204` (`S2` `mmaction/datasets/transforms/pose_transforms.py` L710–991) |
| `T6` | `test_finite_imputation_and_imputation_jump_masking_in_jm_and_bm` | Departure test for finite coordinate imputation (`O2`) and zeroing spurious `jm`/`bm` velocity spikes on imputed-to-real transitions (`FO7`) | Missing/low-conf/`(0,0)` joints are forward-filled (or bbox-centered at frame 0) with `is_confident == False`; when a forward-filled joint snaps to a distant real position at frame `k+1`, `jm[:, k, v, :2] == 0.0` and connected bone motion `bm[:, k, :, :2] == 0.0` | `[UNTESTED DEPARTURE 2]` — grounded in `E04`, `E204`, `O2`, `FO7` |
| `T7` | `test_uniform_sample_100_order_and_format_gcn_m2_zero_pad` | Guard that `GenSkeFeat` executes **before** `UniformSampleFrames(100)` (`np.mod`) and `FormatGCNInput(num_person=2, mode='zero')` (`E204`) | Output tensor has shape `(1, 2, 100, 17, 3)` with person slot `m=1` all zeros; `jm` sampled after `ToMotion` preserves non-zero frame-to-frame velocities across all 100 sampled frames | `E204` (`S2` `mmaction/datasets/transforms/pose_transforms.py` L995–1152 & `formatting.py` L388–445) |
| `T8` | `test_fps_resampling_to_25hz_and_gap_reset_policy` | Departure test for FPS-aware `25.0 Hz` temporal resampling and `> hold_s` (`1.0s`) track-gap buffer reset (`FO7`) | A `12.5 fps` (`dt = 0.08s`) trajectory and a `25.0 fps` (`dt = 0.04s`) trajectory of the same physical motion produce identical resampled `(30, 17, 3)` arrays and `(1, 2, 100, 17, 3)` `j`/`jm` tensors (`atol=1e-4`); a gap `dt = 1.5s > hold_s` returns `None` | `[UNTESTED DEPARTURE 2]` — grounded in `E04`, `E204`, `FO7` |
| `T9` | `test_transfer_gate_evaluates_norm_modes_and_finetune_step` | Pre-state-machine transfer evaluation (`prenormalize2d` vs. `track_compact`) and fine-tuning step (`FO3`) | `evaluate_tsstg_transfer_on_sequences` achieves `fall_recall == 1.0` and `sitting_specificity == 1.0` on real `MMAction2TSSTG` weights; `finetune_tsstg_on_yolo26` reduces `CrossEntropyLoss` over 3 gradient steps without altering buffer `A` | `[UNTESTED DEPARTURE 1]` — grounded in `E01`, `E201`–`E205`, `FO3` |
| `T10` | `test_ref_scale_outlier_clamped_and_non_decaying` | Departure test for outlier-clamped 90th-percentile `ref_scale` (`O3`, `FO5`) | Standing `(50, 180)` and lying `(180, 50)` give equal diagonal; 24s of compressed sitting `(50, 90)` does not decay below `ref_scale_init`; a 2-frame two-person bbox merge spike `(300, 220)` is clamped (`<= 1.15 * median`) | `[UNTESTED DEPARTURE 4]` — fixes `E02` (`S0` L62–63) per `O3` and `FO5` |
| `T11` | `test_track_stitching_and_cold_start_under_30_frames` | Departure test for spatial-temporal track stitching across mid-fall ID switches (`FO6`) and `< 30` frame cold-start fallback (`O5`) | (a) When `tid=1` drops at `t=0.5s` and switches to `tid=2` at `t=0.6s` while lying nearby (`< 0.8 * ref_scale`), `tid=2` inherits `tid=1`'s drop/latch and emits `fall` after `0.6s` stability; (b) a 20-frame (`< 30`) cold-start fall emits `fall` via Path C | `[UNTESTED DEPARTURE 5]` — grounded in `E04`, `E05`, `O5`, `FO6` |
| `T12` | `test_hybrid_fall_latch_requires_descent_and_lying_stability_and_vetoes_seated` | Departure test for Path A & Path B controlled-action/knee-angle veto and strict lying post-fall stability (`FO4`) | (a) `pred_idx == 6` for `2.4s` while seated (`aspect = 0.65`, `angle = 38`, `hip_rel = 0.52`) emits `0` `fall` events; (b) rapid sit-down (`drop > 0.20`, `vdown > 0.30`) vetoed when `MMAction2TSSTG` predicts `'Sit down'` (`A8`/`A6`) or controlled knee bend; (c) true collapse followed within `3.0s` by `0.6s` of horizontal lying emits `1` `fall` candidate | `[UNTESTED DEPARTURE 3]` — grounded in `E01`, `E04`, `E05`, `E205`, `PO2`, `PO3`, `FO4` |
| `T13` | `test_lying_candidate_hold_runs_and_merge_runs_semantics` | Candidate `lying` persistence over `T_lying = 2.0s` with `hold_s = 1.0s` and `cooldown_s = 5.0s` (`FO11`) | Lying state from `t=0.0..1.0s`, a `0.5s` (`<= hold_s`) gap from `1.0..1.5s`, and lying from `1.5..2.2s` bridges to `e - s = 2.2s >= 2.0s` via `hold_runs`/`merge_runs` and emits `type == 'lying'` | `E04`, `E05` (`S0` `fall_monitor.py` L76–117), `FO11` |
| `T14` | `test_windowed_displacement_immobility_ignores_jitter_and_imputed_joints` | Departure test for 2.0s windowed confident-joint excursion (`disp_win`) and posture-differentiated durations (`FO5`) | (a) Stationary person with `1.5 px` random 25 Hz keypoint jitter (`disp_win < 0.02 <= 0.035`) triggers `immobility` at `t >= 5.0s` (post-fall), `10.0s` (lying), and `120.0s` (seated, with `0` alerts at `30s`); (b) moving person whose 3 confident joints move while 14 occluded joints are forward-filled does NOT trigger false immobility | `[UNTESTED DEPARTURE 4]` — grounded in `E02`, `E03`, `O9`, `FO5` |
| `T15` | `test_candidate_lifecycle_dedup_event_evidence_and_cli_run` | Contract test for AC #4 (`event_id`, `review_event` states), `cooldown_s` vs. `realert_s` deduplication (`E05`), structured `evidence` dict, and `run()` video pipeline integration (`FO9`, `FO11`) | Emitted event has `event_id`, `status == 'candidate'`, and `evidence` (`clip_window`, `tsstg_probs`, `ntu60_top5`, `kinematics`, `snapshot_path`); `review_event(event_id, 'confirmed', 'verified')` updates state; synthetic `.mp4` run writes `.jsonl` events | `[PROJECT-SPECIFIC]` — `problem.md` AC #4 & §5, `E05`, `FO9`, `FO11` |
| `T16` | `test_le2i_annotation_parser_and_heldout_benchmark_metrics` | Contract test for Le2i ground-truth parser and held-out evaluation metrics (`precision`, `recall`, `f1`, `FA/h`, `latency_s`, `lookalike_fa_breakdown`) (`FO10`) | `parse_le2i_annotation` parses fall frame intervals at 25 fps; `compute_benchmark_metrics` computes exact Precision, Recall, F1, FA/h, mean latency, and per-activity look-alike false alarms (`bending`, `kneeling`, `sitting`, `floor_work`) | `E01`, `E04`, `E205`, `problem.md` AC #5 & #6, `FO10` |
