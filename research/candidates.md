# Phase 4: Synthesis, Ranking of Candidate Solutions, and Understanding Report

## 1. The Problem as Understood After Full Literature & Code Inspection

Our objective is to build an accurate, real-time candidate detection pipeline on top of frozen **Ultralytics `yolo26x-pose.pt` (17 COCO keypoints) + `ByteTrack`** starting from [`fall_monitor.py`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py) (`S0`) that satisfies all six Acceptance Criteria:
1. **Candidate fall events** (`fall`).
2. **Candidate lying-person events** (`lying` persisting for a configured duration).
3. **Candidate prolonged unusual immobility events** (`immobility` when a tracked person exhibits little observable movement over a configured duration).
4. **Distinguishing candidate detections from confirmed incidents** (preserving operator review metadata & event evidence).
5. **Suppressing look-alike activities** (bending, kneeling, sitting, ordinary floor work).
6. **Evaluated on representative video sequences** (Le2i + custom CCTV clips) with documented performance and limitations.

### Key Findings on the Baseline (`S0`: [`fall_monitor.py`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py))
- `[SOURCE: S0, fall_monitor.py, line 58 (E01)]` [`compute_features`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py#L53-L74) only extracts head (`[0, 1, 2]`), shoulders (`[5, 6]`), and hips (`[11, 12]`), completely ignoring **knees (`[13, 14]`) and ankles (`[15, 16]`)**. Because lower-limb keypoints are unused, the baseline cannot observe whether the person's legs are vertically supporting them (bending/crouching), folded in a chair (sitting), or kneeling.
- `[SOURCE: S0, fall_monitor.py, lines 62-63 (E02)]` Reference height `ref_h` is computed over a rolling `10000ms` (10-second) window (`rolling('10000ms', min_periods=1).quantile(0.9)`), and `hip_rel` is measured from the current frame's bounding box bottom `box[:, 3]`. If a person sits, crouches, or lies down for more than 10 seconds, `ref_h` shrinks to the compressed bounding-box height, distorting `hip_rel`, `vdown`, and `motion`.
- `[SOURCE: S0, fall_monitor.py, lines 110-141 & 165 (E03)]` `fall_monitor.py` implements [`detect_falls`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py#L119-L141) and [`detect_lying`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py#L110-L117), but contains **no detector for prolonged unusual immobility**.

### Key Findings on Priority 1 Open-Source Temporal Models (`S1`, `S2`, `S12`)
During Phase 2 & 3, we confirmed that **two complete, open-source Priority 1 repositories with downloadable pretrained `.pth` weights on 2D COCO keypoints exist**:
1. **`S12` (`GajuuzZ/Human-Falling-Detect-Tracks`) — Pretrained Two-Stream ST-GCN (`tsstg-model.pth`) Trained Specifically on Le2i**:
   - `[SOURCE: S12, ActionsEstLoader.py, class TSSTG.__init__ (E101)]` Classifies 30-frame skeleton windows into 7 exact classes relevant to our task: `['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']`.
   - `[SOURCE: S12, PoseEstimateLoader.py & ActionsEstLoader.py (E102, E103)]` Drops the 4 eye/ear keypoints (`1, 2, 3, 4`) from the 17 COCO keypoints (`[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]` $\rightarrow$ 13 keypoints) and appends the shoulder midpoint `(kp[5] + kp[6]) / 2` as node 13 (`Center-Shoulder`), forming a 14-node `coco_cut` graph. Directly compatible with `yolo26x-pose.pt`'s 17 COCO keypoints without retraining!
   - `[SOURCE: S12, ActionsEstLoader.py & Actionsrecognition/Models.py (E104, E105)]` Takes a position stream `pts` of shape `(1, 3, 30, 14)` and a motion stream `mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]` of shape `(1, 2, 29, 14)`. However, `scale_pose` scales each frame independently to `[-1, 1]^2` (`((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1`), which **destroys bounding-box aspect ratio** (`w/h`) and true body-height drop scale.
2. **`S2` (`kennymckormick/pyskl`, Duan et al. CVPR 2022) — Pretrained ST-GCN / ST-GCN++ / PoseConv3D on 17 COCO Keypoints**:
   - `[SOURCE: S2, configs/stgcn/README.md & arXiv:2104.13586 Table 2 (E106, E107)]` Provides downloadable `.pth` weights trained on 17 COCO 2D keypoints for NTU-60/120: `ST-GCN` (Joint `89.0%`, Bone `91.2%`, 2-Stream `92.0%`, 4-Stream `92.4%`) and `PoseConv3D SlowOnly-R50` (`93.7%` / `94.1%` Joint+Limb).
   - `[SOURCE: S2, tools/data/label_map/nturgbd_120.txt line 43 & pose_related.py (E109a, E109b)]` Class index `42` (Action `A43`) is `"falling down"`, and `PreNormalize2D` normalizes coordinates relative to full frame `(w/2, h/2)` rather than per-frame min-max.
   - `[SOURCE: S2, arXiv:2104.13586 Section 4.2 Table 4 (E108)]` Demonstrates a major limitation of coordinate-based GCNs: when 1 limb keypoint is dropped ($p=1$), coordinate GCN (`MS-G3D`) accuracy plunges by **14.3 percentage points** (`92.0%` $\rightarrow$ `77.7%`), whereas 3D heatmap volume `PoseConv3D` drops by only **0.9 percentage points** (`92.4%` $\rightarrow$ `91.5%`).

---

## 2. Ranked Candidate Solutions

We score each candidate across 5 explicit criteria (0–5 each, max 25):
1. **Fit to User Constraints** (works on `yolo26x-pose.pt` + `ByteTrack`, detects Fall + Lying + Immobility, handles bending/kneeling/sitting/floor work).
2. **Strength & Independence of Evidence** (backed by verified `FULL`-read sources).
3. **Reproducibility & Ready-to-Use Code/Weights** (public GitHub code + weights matching our 17 COCO keypoints).
4. **Simplicity & Low Latency** (minimal overhead on top of `yolo26x-pose.pt`).
5. **Low Risk / Failure Resilience** (handles missing/occluded keypoints and aspect-ratio preservation).

| Rank | Candidate Architecture | Fit (5) | Evidence (5) | Reprod. (5) | Simplicity (5) | Low Risk (5) | **Total (25)** | Summary |
|---|---|---|---|---|---|---|---|---|
| **1 (Tied / Staged)** | **Candidate A (Stage 1 Demo): Biomechanical & Kinematic Multi-Condition State Machine on 17 COCO Keypoints** (Upgraded `fall_monitor.py` adding lower-limb `[13..16]` geometry, non-decaying upright `ref_h`, look-alike overrides, and `detect_immobility`) | **5** | **4** | **5** | **5** | **4** | **23 / 25** | Zero extra neural weights; fixes all 3 structural bugs in `fall_monitor.py` (`E01`–`E03`); immune to `scale_pose` aspect distortion (`E104`) and GCN keypoint-dropout collapse (`E108`). |
| **1 (Tied / Staged)** | **Candidate B (Stage 2 Hybrid): Upgraded Kinematic Rules + Pretrained `TSSTG` (`S12`, `tsstg-model.pth`) Two-Stream ST-GCN Ensemble/Gating** | **5** | **5** | **5** | **4** | **4** | **23 / 25** | Combines Candidate A's aspect-preserving kinematic state machine with `S12`'s pretrained 30-frame 14-node `coco_cut` ST-GCN (`Fall Down`, `Lying Down`, `Sitting`, `Standing`, `Walking`) `[INFERENCE: C8]`. |
| **3** | **Candidate C: Standalone Pretrained `TSSTG` (`S12`, `GajuuzZ/Human-Falling-Detect-Tracks`) Replacing Kinematic Rules** | **3** | **4** | **5** | **4** | **2** | **18 / 25** | Has Le2i pretrained weights (`E101`), but `scale_pose` destroys bbox aspect ratio (`E104`),coordinate GCNs degrade by 14.3 pp under keypoint occlusion (`E108`), and lacks a prolonged `immobility` class. |
| **4** | **Candidate D: `PYSKL` (`S2`) Pretrained `ST-GCN++` / `PoseConv3D` (`ntu60_xsub_hrnet`) on 17 COCO Keypoints** | **3** | **5** | **5** | **2** | **4** | **19 / 25** | Strongest general action recognition accuracy (`92.4%`–`94.1%` on NTU-60, `E106`–`E107`) and occlusion robustness (`PoseConv3D`, `E108`), but NTU-60 only has `A43 = falling down` (no persistent `Lying Down` class) and `PoseConv3D` adds 3D-CNN heatmap volume overhead (`15.9G` FLOPs). |
| **5** | **Candidate E: Custom-Trained Hypergraph Mamba (`S4` FLASH) or 3-Stream ST-GCN (`S3`)** | **2** | **2** | **1** | **1** | **2** | **8 / 25** | Neither `S3` nor `S4` releases runnable training/model code or pretrained `.pth` weights (`S4` repo only has `plot_hypermamba_arch.py`), and `S4` requires 33 3D BlazePose joints rather than 17 2D COCO keypoints. |

> **Why Candidate A (Stage 1) + Candidate B (Stage 2) is Optimal**:
> As established in Claim `C8` `[INFERENCE: E01, E02, E104, E105, E109b]`, `S0` (`fall_monitor.py`) and `S12` (`TSSTG`) have **exact complementary strengths and blind spots**:
> - `S0` preserves true image aspect ratio (`w/h`) and body-height-normalized drop velocity (`vdown`), handles missing keypoints gracefully via `NaN` hysteresis (`hold_runs`), and supports arbitrary duration thresholds for `lying` and `immobility`—but ignores lower-body keypoints `[13..16]` and suffers from 10s `ref_h` decay.
> - `S12` (`tsstg-model.pth`) uses all 13 body keypoints (including knees `[13, 14]` and ankles `[15, 16]`) + neck `[13]` over 30 frames to classify `Sitting`, `Standing`, `Walking`, `Sit down`, `Stand up`, `Lying Down`, and `Fall Down`—but its per-frame `scale_pose` stretches every bounding box to `[-1, 1]^2` (erasing aspect ratio and absolute drop height) and it cannot detect prolonged immobility in a seated/standing posture.

---

## 3. Conflicts Between Sources

1. **ST-GCN Layer Count & Adjacency Normalization (`S1` Paper vs. `S1` Code vs. `S12` Code)**:
   - `[SOURCE: S1, arXiv:1801.07455 Section 3.3 & Eq. 3 vs. yysijie/st-gcn net/st_gcn.py & net/utils/graph.py (E111a, E111b)]` The AAAI 2018 paper states ST-GCN has **9 layers** and uses **symmetric normalization** $\mathbf{\Lambda}^{-1/2}\mathbf{A}\mathbf{\Lambda}^{-1/2}$. However, `yysijie/st-gcn` implements **10 `st_gcn` layers** and uses **asymmetric random-walk normalization** $\mathbf{A}\mathbf{D}^{-1}$ in `normalize_digraph`. Meanwhile, `S12` (`GajuuzZ/Human-Falling-Detect-Tracks`, `Actionsrecognition/Utils.py`) also uses **10 `st_gcn` layers** but implements **symmetric normalization** $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ in its `normalize_digraph`.
   - *Impact on Our Pipeline*: When loading `S12`'s pretrained `tsstg-model.pth` in Stage 2, we **must** use `S12`'s exact 10-layer architecture and symmetric `normalize_digraph` (`D^{-1/2} A D^{-1/2}`), whereas when loading `S2`/`S1` checkpoints we must match their respective graph normalization.
2. **Coordinate Normalization (`S12` Per-Frame Box Scaling vs. `S2` Full-Frame Centering)**:
   - `[SOURCE: S12, pose_utils.py scale_pose (E104) vs. S2, pyskl pose_related.py PreNormalize2D (E109b)]` `S12` normalizes each frame's skeleton using that frame's `nanmin`/`nanmax` (`[-1, 1]^2`), whereas `S2` (`PYSKL`) normalizes using the full image dimensions `(w/2, h/2)`, preserving the aspect ratio of the pose.
3. **GCN vs. 3D Heatmap Robustness to Missing Keypoints (`S2`)**:
   - `[SOURCE: S2, arXiv:2104.13586 Table 4 (E108)]` Coordinate GCNs lose `14.3` percentage points when a limb keypoint is dropped, whereas `PoseConv3D` loses `0.9` percentage points.

---

## 4. Sources Not Read in Full (`PARTIAL` / `BLOCKED`) and Verification Rejection Counts

### A. Sources Excluded from the Verified `evidence.jsonl` Base (`read_status: PARTIAL`)
Per Rule 4 of `ai-researcher` and Claim Check Rule 2 of `references/evidence-ledger.md`, the following 10 sources were marked `PARTIAL` (because full PDFs or raw files were not downloaded locally or were behind paywalls/missing from repos) and kept out of `evidence.jsonl`:
- `S3` (`arXiv:2408.12211`, Shin et al. 2024/2025 Three-Stream ST-GCN) — `PARTIAL` (no public code/weights; abstract/summary verified).
- `S4` (`arXiv:2607.25791`, Koffi et al. 2026 FLASH Hypergraph-Mamba) — `PARTIAL` (GitHub repo `Tresor-Koffi/FLASH-Impact-Fall-Detection` contains only `README.md` and `plot_hypermamba_arch.py`, no runnable code or weights).
- `S5` (`arXiv:2502.19048`, Koffi et al. 2025 Improved 3D Skeletons UP-Fall Dataset) — `PARTIAL` (GitHub repo `Tresor-Koffi/3D_skeletons-UP-Fall-Dataset` contains `README.md` and Zenodo CSV link, no training code or weights).
- `S6` (`DOI:10.3390/s25216783`, Zobi et al. 2025 VIRA-GCN) — `PARTIAL` (no public code/weights).
- `S7` (`arXiv:2509.05337`, Cho et al. 2025 DGNN-LSTM) — `PARTIAL` (no public code/weights).
- `S8` (`arXiv:1908.00275`, Hua et al. 2019 Seq2Seq) — `PARTIAL` (no public code/weights).
- `S9` (`DOI:10.3390/info17100976`, Alam et al. 2026 YOLO12n+MediaPipe 4 Overrides) — `PARTIAL` (no public code; internal threshold numbers not indexed in web search).
- `S10` (`arXiv:2503.19501`, Mali & Jaiswal 2025) — **REJECTED (`does_not_support_claim`)**: **Withdrawn on arXiv by the authors due to `"Misleading Results"`**.
- `S11` (`DOI:10.1038/s41598-025-86429-6`, `PMC11751301`, Liu et al. 2025 HFDMIA-Pose) — `PARTIAL` (internal threshold numbers not extracted locally).
- `S13` (`cwlroda/falldetection_openpifpaf`) — `PARTIAL` (README verified; admits false positives and true negatives were not tested on URFD).
- `S14` (`DOI:10.1117/1.JEI.22.4.041106`, Charfi et al. 2013 Le2i Dataset) — `PARTIAL` (SPIE paywall).

### B. Verification & Rejection Counts (Phase 1 Triage + Phase 2/3 Skeptic Audit)
- **Phase 1 Triage Rejections**: **8 sources rejected** (`R1`–`R4`: `metadata_wrong`; `R5`–`R7`: `does_not_support_claim`; `S10`: withdrawn due to `"Misleading Results"` $\rightarrow$ `does_not_support_claim`).
- **Phase 2/3 Independent Skeptic Audit of 32 Proposed `evidence.jsonl` Entries**:
  - **14 `CONFIRMED`** (expanded to **17 atomic `supported` records** `E01`–`E03`, `E101`–`E108`, `E109a`–`E111b` after splitting compound claims `E109`, `E110`, `E111`).
  - **3 `PARTIAL`** (`E112`, `E113`, `E114` from `S4`/`S5` — caught derived-value arithmetic error in `E113` where memory reduction percentages `47.1%` and `56.6%` had been applied to FLOPs; removed from `evidence.jsonl` when `S4`/`S5` were downgraded to `PARTIAL`).
  - **14 `BLOCKED`** (`E_S9_1`–`E_S14_1` from `PARTIAL` sources `S3, S6, S9, S11, S13, S14` — removed from `evidence.jsonl` to satisfy Claim Check Rule 2).
  - **1 `REJECTED`** (`E_S10_1` from withdrawn preprint `S10` + parenthetical numbers in `E113`).

---

## 5. Bibliography of Verified `FULL`-Read Sources (`read_status: FULL`)

1. **`S0`**: [`fall_monitor.py`](file:///c:/Users/Maha%20Mamdouh/Downloads/fall_monitor.py) — Local baseline pose-based FALL / LYING candidate detector using Ultralytics `yolo26x-pose.pt` + `ByteTrack` (2026). Notes: [`S0.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/sources/S0.md).
2. **`S1`**: Sijie Yan, Yuanjun Xiong, Dahua Lin (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018 (`arXiv:1801.07455`, https://arxiv.org/abs/1801.07455). Official Repo: https://github.com/yysijie/st-gcn. Notes: [`S1.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/sources/S1.md).
3. **`S2`**: Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, Bo Dai (2022). *Revisiting Skeleton-based Action Recognition*. CVPR 2022 (`arXiv:2104.13586`, https://arxiv.org/abs/2104.13586). Official Repo: https://github.com/kennymckormick/pyskl. Notes: [`S2.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/sources/S2.md).
4. **`S12`**: GajuuzZ (2020). *Human-Falling-Detect-Tracks* (`TwoStreamSpatialTemporalGraph` / `tsstg-model.pth` on Le2i). Official Repo: https://github.com/GajuuzZ/Human-Falling-Detect-Tracks. Notes: [`S12.md`](file:///C:/Users/Maha%20Mamdouh/.gemini/antigravity/scratch/fall_detection_research/research/sources/S12.md).
