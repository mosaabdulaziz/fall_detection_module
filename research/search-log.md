# Search & Triage Log (Phase 1)

**Date Retrieved from Environment**: `2026-10-21`
**Depth Mode**: `standard` (8–15 sources to be read in full + GitHub code verification)

## Search Angles & Queries Executed

1. **Angle 1 — Task + YOLO-pose + Benchmarks + GitHub**:
   - Query: `"fall detection" ("YOLOv8-pose" OR "YOLO-pose" OR "YOLOv11-pose" OR "keypoint") "github.com" ("Le2i" OR "UR Fall" OR "UP-Fall")`
2. **Angle 2 — Two-Stage Skeleton Action Recognition (Priority 1)**:
   - Query: `"fall detection" skeleton "ST-GCN" OR "TCN" OR "LSTM" OR "Transformer" "COCO" keypoints "github.com" arxiv`
   - Query: `site:arxiv.org "fall detection" ("2D pose" OR "skeleton" OR "keypoints") ("ST-GCN" OR "LSTM" OR "GRU" OR "Transformer" OR "TCN") ("Le2i" OR "UP-Fall" OR "URFD" OR "UR Fall")`
   - Query: `site:arxiv.org "Revisiting Skeleton-based Action Recognition" "2104.13586" OR "PoseConv3D" "fall" OR "1801.07455"`
   - Query: `site:arxiv.org "fall detection" "github.com" ("pose" OR "skeleton" OR "keypoint")`
3. **Angle 3 — Biomechanical & Kinematic Pose Rules + False-Positive Suppression (Priority 2)**:
   - Query: `"fall detection" pose estimation rule-based OR threshold "bending" "sitting" "kneeling" angle velocity aspect ratio arxiv`
   - Query: `site:arxiv.org "fall detection" ("rule-based" OR "threshold" OR "angle" OR "aspect ratio") ("pose estimation" OR "skeleton") ("false positive" OR "bending" OR "sitting")`
   - Query: `"Vertical Leg Override" OR "Aspect Ratio Override" "fall detection" mdpi OR arxiv OR github`
   - Query: `"A Pose Landmark-Based Lightweight Human Fall-Detection Framework Using Spatiotemporal Features" authors year doi mdpi github`
4. **Angle 4 — Critiques, Failure Modes, Cross-Dataset & Occlusion/Data Quality Limits (Negative Evidence)**:
   - Query: `"fall detection" ("false positive" OR "failure" OR "limitation" OR "critique" OR "cross-dataset") "skeleton" OR "pose" ("Le2i" OR "UP-Fall" OR "UR Fall") site:arxiv.org OR site:mdpi.com OR site:pmc.ncbi.nlm.nih.gov`
   - Query: `"arXiv:2502.19048" OR "2502.19048" "An Improved 3D Skeletons UP-Fall Dataset" github OR Tresor Koffi`
   - Query: `"VIRA-GCN" "fall detection" OR "skeleton" mdpi OR arxiv OR doi`
5. **Angle 5 — Metadata & Fabrication Verification**:
   - Query: `"2503.19501" OR "2503.04505" OR "2408.12154" OR "2502.19048" OR "2409.02104" arxiv`
   - Query: `"2408.12211" OR "2509.05337" OR "2607.25791" arxiv title authors`
   - Query: `"PMC11749004" OR "s41598-025-87383-z" "fall detection" github OR method OR angle OR velocity`

---

## Rejected Candidates (Logged per `search-playbook.md` & `rejection-taxonomy.md`)

1. **`R1` — `arXiv:2408.12154` (initially surfaced by web summary as Three-Stream ST-GCN)**
   - **Reason**: `metadata_wrong` (Resolved to *Binary codes from subset inclusion matrices* by Marin & Mogilnykh in `math.CO`; the actual Three-Stream ST-GCN paper is `arXiv:2408.12211`, registered as `S3`).
2. **`R2` — `arXiv:2503.04505` (initially surfaced by web summary as MediaPipe Fall Detection)**
   - **Reason**: `metadata_wrong` / `does_not_support_claim` (Resolved to *Tuning Magnetism of Metal-Organic Framework by Different Types of Pressure* in Condensed Matter; rejected).
3. **`R3` — `arXiv:2409.02104` (initially surfaced by web summary as a Fall Detection Review)**
   - **Reason**: `metadata_wrong` / `does_not_support_claim` (Resolved to *DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction* by Seidenschwarz et al.; rejected).
4. **`R4` — `PMC11749004` / `10.1038/s41598-025-87383-z` (initially surfaced as Hybrid YOLOv8s + AlphaPose)**
   - **Reason**: `metadata_wrong` (Resolved to unrelated proteomics / chemistry articles; the actual Liu et al. 2025 paper is `10.1038/s41598-025-86429-6` / `PMC11751301`, registered as `S10`).
5. **`R5` — `arXiv:2408.04605` (*Fall Detection for Industrial Setups Using YOLOv8 Variants*, Pereira 2024)**
   - **Reason**: `does_not_support_claim` (Treats fall detection purely as single-frame 2D bounding-box object detection (`yolov8n/s/m`) without pose keypoints or temporal tracking; fails our pose + temporal scope).
6. **`R6` — `DesertDemons/Fall_Detection_Template_Using_YOLOv8-pose` (GitHub)**
   - **Reason**: `tertiary` / `does_not_support_claim` (Unreviewed student template using single-frame head-to-hip proximity without temporal tracking, velocity, or benchmark evaluation).
7. **`R7` — `Varuni08/Fall-Detection-System` (GitHub)**
   - **Reason**: `tertiary` (Basic single-frame YOLOv8-pose threshold demo focused on ONNX runtime speed test rather than validated temporal/kinematic rules or multi-class ADL separation).
