# Pose-Based Human Fall, Persistent Lying, and Immobility Detection Pipeline

A clinical-grade computer vision and spatiotemporal graph convolutional pipeline for real-time human fall detection, prolonged floor recumbence (persistent lying), and post-fall motionless triage (immobility monitoring).

Designed for elderly care facilities, assisted living communities, and hospital rooms using standard single-camera video streams without wearable sensors.

---

## 1. System Architecture Overview

The system processes standard video streams (RGB, default $`320 \times 240`$ or $`640 \times 480`$ at $`25.0\,\text{Hz}`$) through a six-stage decoupled architecture:

```text
[Camera Stream / Video File / RTSP]
            │
            ▼
┌───────────────────────────────┐
│ 1. Person Detection & Tracking│  YOLOv26x-Pose (COCO 17 Keypoints) + ByteTrack
│    + Spatial-temporal Stitch  │  Tracks stitched across occlusions <= 3.0s
└──────────────┬────────────────┘
               │ 17-Keypoint Coordinates (x, y, conf) + Bounding Boxes
               ▼
┌───────────────────────────────┐
│ 2. Feature Adapter & Resample │  25 Hz Uniform Resampling + Causal Imputation
│    + Kinematic Extraction     │  Imputation-jump masking at boundary gaps
└──────────────┬────────────────┘
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
┌─────────────────────────┐   ┌───────────────────────────┐
│ 3. Multi-Stream ST-GCN  │   │ 4. Geometric & Kinematic  │
│    (Joint + Joint-Motion│   │    Collapse Analysis      │
│    2.5s / 100 Frames)   │   │    (Drop, V_down, Angle)  │
└──────────────┬──────────┘   └─────────────┬─────────────┘
               │ P(Actions)                 │ Features
               └───────┬────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Triangulated Decision Engine & Post-Fall Confirmation│
│    • Path A: High-Confidence ST-GCN Impact Trigger      │
│    • Path B: Kinematic Collapse + Action Support        │
│    • Path C: Cold-Start / Pre-Window Geometric Fallback │
│    • Controlled Vetoes: Crouch, Knee Bend, Upright Stance│
│    • Multi-Mode Posture Confirmation (Recumbent/Curled) │
│    • Scale-Normalized Immobility Window Analysis        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Two-Tier Event Lifecycle & Alarm Dispatch            │
│    • 'fall': Confirmed fall with sustained recumbence   │
│    • 'fall_recovered': Self-recovery within post window │
│    • 'lying': Prolonged floor posture without impact    │
│    • 'immobility': Post-fall stillness / floor triage   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Innovations & Mathematical Formulations

### 2.1 Decoupled Single Source of Truth

All thresholds, physical limits, video dimensions, sampling rates, and semantic mappings are strictly decoupled into `config.yaml`. The codebase contains zero hardcoded magic numbers, and modifying `config.yaml` dynamically controls system decision boundaries.

### 2.2 8-Class Semantic Action Taxonomy (`ActionClass`)

To eliminate false alarms caused by normal walking keypoint jitter at low resolution, NTU-RGB+D action classes are partitioned into an 8-class semantic space:

| Index | Semantic Class | Source NTU Classes   | Description                                         |
|:-----:|:---------------|:---------------------|:----------------------------------------------------|
| **0** | `Standing`     | A1                   | Upright stationary stance                           |
| **1** | `Walking`      | A26, A27, A59, A60   | Normal locomotion (**A42 excluded**)                |
| **2** | `Sitting`      | A11, A12, A29, A30   | Seated upright                                      |
| **3** | `Lying`        | A43 (terminal)       | Flat recumbent posture on floor or bed              |
| **4** | `StandUp`      | A9                   | Controlled transition from seated/lying to standing |
| **5** | `SitDown`      | A6, A8, A16, A17     | Controlled descent to chair or bench                |
| **6** | `Fall`         | A43 (dynamic impact) | Sudden uncontrolled descent                         |
| **7** | `Instability`  | **A42 (staggering)** | Balance perturbation / stumbling                    |

> [!NOTE]
> Population evaluation across 2,172 frames proved that $`P(\text{A43})`$ separates falls with an AUROC of **0.8858**, whereas $`P(\text{A42})`$ has an AUROC of **0.6444** and spikes up to **0.9983** during normal walking due to low-resolution keypoint noise. Isolating A42 into class 7 prevents walking jitter from triggering false alarms while allowing instability to serve as soft auxiliary evidence for Path B.

### 2.3 Triangulated Latch Decision Paths

A candidate fall latch is triggered if **any** of three complementary paths fire.

**Path A (Classifier Trigger)**

```math
\text{has\_win} \land \text{pred} = 6 \land P_{\text{smooth}}(\text{Fall}) \ge 0.30 \land \neg\text{veto} \land \left(\text{drop} \gt 0.08 \lor v_{\text{down}} \gt 0.12\right) \land \left(\theta \gt 40^\circ \lor \text{aspect} \gt 0.85 \lor \text{hip}_{\text{rel}} \lt 0.35\right)
```

**Path B (Kinematic Collapse with Action Support)**

```math
\begin{aligned}
&\text{has\_win} \land \text{drop} \gt 0.14 \land v_{\text{down}} \gt 0.16 \land \neg\text{veto} \land \text{hip}_{\text{rel}} \lt 0.35 \\
&\land \Big( P_{\text{smooth}}(\text{Fall}) \ge 0.15 \lor \left(P_{\text{smooth}}(\text{Lying}) + P_{\text{smooth}}(\text{Fall}) \ge 0.25\right) \lor P_{\text{smooth}}(\text{Instability}) \ge 0.35 \Big)
\end{aligned}
```

**Path C (Cold-Start / Missing Model Geometric Fallback)**

```math
\neg\text{has\_win} \land \text{drop} \gt 0.14 \land v_{\text{down}} \gt 0.16 \land \text{hip}_{\text{rel}} \lt 0.32 \land \left(\theta \gt 45^\circ \lor \text{aspect} \gt 1.05 \lor \text{head}_{\text{rel}} \lt 0.0\right)
```

### 2.4 Controlled-Action Vetoes

Before latching a fall, the system evaluates biomechanical criteria to veto deliberate activities of daily living:

1. **Controlled Bend**: $`\theta \lt 45^\circ \land \text{aspect} \lt 0.85 \land \text{head}_{\text{rel}} \gt 0.30 \land v_{\text{down}} \le 0.30`$
2. **Controlled Knee Bend**: $`70^\circ \le \theta_{\text{knee}} \le 110^\circ \land \theta \lt 45^\circ \land \text{aspect} \lt 0.85 \land \text{hip}_{\text{rel}} \ge 0.35`$
3. **Upright Controlled Action**: Upright posture ($`\theta \lt 45^\circ,\ \text{aspect} \lt 0.85,\ \text{head}_{\text{rel}} \gt 0.20`$) where upright probabilities (Standing, Walking, Sitting, SitDown, StandUp) exceed fall/lying probabilities.

### 2.5 Multi-Mode Post-Fall Posture Confirmation

Once latched, the pipeline monitors the trailing window ($`T_{\text{post}} \le 3.0\,\text{s}`$) to confirm genuine incapacitation, accommodating three distinct clinical end-states:

1. **Mode 1 (Standard Horizontal Recumbence)**

   ```math
   \text{hip}_{\text{rel}} \le 0.35 \land \left(\theta \ge 40.0^\circ \lor \text{aspect} \ge 0.90\right)
   ```

2. **Mode 2 (Floor-Curled / Seated Collapse)**

   ```math
   \text{hip}_{\text{rel}} \le 0.28 \land \text{drop} \ge 0.10 \land \left(\theta \ge 20.0^\circ \lor \text{aspect} \ge 0.85 \lor \text{head}_{\text{rel}} \le 0.12\right)
   ```

3. **Mode 3 (Low-Seated / Sofa / Bed Slump)**

   ```math
   \text{hip}_{\text{rel}} \le 0.40 \land \text{drop} \ge 0.14 \land \left(\theta \ge 35.0^\circ \lor \text{aspect} \ge 0.90\right)
   ```

Additional guards:

- **Upright Chair Sitting Rejection**: Explicitly rejects sitting false alarms if $`\theta \lt 20.0^\circ \land \text{aspect} \lt 0.70`$.
- **Motion Guard**: Posture must hold for $`\ge 0.50\,\text{s}`$ with scale-normalized motion $`\le 1.80`$ (allowing post-impact tremor or arm movement while rejecting active walking recovery).

### 2.6 Immobility Detection via Windowed Centroid Excursion

To detect post-fall shock or acute medical collapse, the pipeline computes the median centroid displacement across all confident joints over a sliding temporal window ($`T_{\text{win}}`$):

```math
\text{disp\_win} = \frac{\operatorname*{median}\limits_{j \in \text{valid}} \left( \max\limits_{t} \left\lVert p_j(t) - \bar{p}_j \right\rVert_2 \right)}{\max\left(S_{\text{ref}},\ 1.0\right)}
```

where $`S_{\text{ref}}`$ is the clamped bounding-box diagonal. If $`\text{disp\_win} \le \text{disp\_thresh}`$, the stillness timer accumulates towards dispatch.

---

## 3. Master Tuned Configuration Reference

All system parameters are defined in `config.yaml`:

| Section                      | Parameter                                | Tuned Value      | Unit / Range | Clinical Description                                              |
|:-----------------------------|:-----------------------------------------|:-----------------|:-------------|:------------------------------------------------------------------|
| **`video`**                  | `default_width`                          | `320`            | px           | Base frame width                                                  |
|                              | `default_height`                         | `240`            | px           | Base frame height                                                 |
|                              | `target_fps`                             | `25.0`           | Hz           | Target temporal rate                                              |
| **`detector`**               | `conf_det`                               | `0.25`           | $`[0, 1]`$   | YOLO person detection confidence                                  |
|                              | `imgsz`                                  | `640`            | px           | Detector inference resolution                                     |
|                              | `streams`                                | `['j', 'jm']`    | list         | Active ST-GCN input modalities (joint + joint-motion)             |
| **`sampling`**               | `conf_kp`                                | `0.30`           | $`[0, 1]`$   | Keypoint confidence filter threshold                              |
|                              | `smooth_s`                               | `0.40`           | s            | Temporal smoothing window for action probabilities                |
|                              | `buffer_s`                               | `30.0`           | s            | Rolling track history retention                                   |
|                              | `eval_every_s`                           | `0.50`           | s            | Periodic evaluation stride                                        |
|                              | `ref_scale_max_growth`                   | `1.15`           | factor       | Maximum growth over median bounding-box diagonal                  |
|                              | `ref_scale_max_history`                  | `1500`           | frames       | Diagonal history length cap                                       |
|                              | `probs_hist_retention_s`                 | `1.0`            | s            | Rolling buffer duration for action probabilities                  |
| **`windowing`**              | `primary_window_s`                       | `2.50`           | s            | Primary ST-GCN temporal clip duration                             |
|                              | `secondary_window_s`                     | `1.25`           | s            | Fast-response secondary window                                    |
|                              | `min_window_s`                           | `1.16`           | s            | Minimum window required before ST-GCN inference                   |
|                              | `clip_len`                               | `100`            | frames       | Resampled uniform frame count fed to ST-GCN                       |
|                              | `max_gap_s`                              | `1.0`            | s            | Maximum track gap tolerated before reset                          |
|                              | `min_observations_for_resample`          | `6`              | frames       | Minimum observations required to resample                         |
| **`normalization`**          | `mode`                                   | `prenormalize2d` | enum         | Scene-relative coordinate normalization mode                      |
|                              | `track_compact_scale_factor`             | `1.25`           | factor       | Aspect ratio scale factor for `track_compact` mode                |
| **`action_classes`**         | `projection.fall_sources`                | `[42]`           | list         | NTU-60 falling (A43) mapped to Fall (idx 6)                       |
|                              | `projection.instability_sources`         | `[41]`           | list         | NTU-60 staggering (A42) mapped to Instability (idx 7)             |
| **`kinematic_veto`**         | `controlled_bend.max_angle`              | `45.0`           | deg          | Maximum torso angle for deliberate bending                        |
|                              | `controlled_knee.min_knee_angle`         | `70.0`           | deg          | Lower knee angle limit for squats/crouches                        |
|                              | `controlled_knee.max_knee_angle`         | `110.0`          | deg          | Upper knee angle limit for squats/crouches                        |
| **`path_a`**                 | `min_action_score`                       | `0.30`           | $`[0, 1]`$   | ST-GCN $`P(\text{Fall})`$ trigger threshold                       |
|                              | `min_drop`                               | `0.08`           | scale        | Minimum vertical hip drop required                                |
|                              | `min_vdown`                              | `0.12`           | scale/s      | Minimum peak downward velocity                                    |
| **`path_b`**                 | `drop`                                   | `0.14`           | scale        | Vertical hip drop required for kinematic collapse                 |
|                              | `vdown`                                  | `0.16`           | scale/s      | Peak downward velocity required                                   |
|                              | `support_fall_score`                     | `0.15`           | $`[0, 1]`$   | Minimum $`P(\text{Fall})`$ to support kinematic drop              |
|                              | `combined_support_score`                 | `0.25`           | $`[0, 1]`$   | Combined $`P(\text{Fall}) + P(\text{Lying})`$ support threshold   |
|                              | `support_instability_score`              | `0.35`           | $`[0, 1]`$   | Minimum $`P(\text{Instability})`$ to support kinematic drop       |
| **`path_c`**                 | `drop` / `vdown`                         | `0.14` / `0.16`  | scale        | Cold-start vertical drop & velocity thresholds                    |
|                              | `low_hip`                                | `0.32`           | scale        | Cold-start vertical hip height threshold                          |
| **`post_fall_confirmation`** | `post_s`                                 | `3.00`           | s            | Post-impact search window duration                                |
|                              | `hold_s`                                 | `1.00`           | s            | Gap tolerance between recumbent detections                        |
|                              | `stable_s`                               | `0.50`           | s            | Required continuous recumbent hold duration                       |
|                              | `max_motion`                             | `1.80`           | scale/s      | Maximum allowed body movement during confirmation                 |
|                              | `std_low_hip`                            | `0.35`           | scale        | Mode 1: Maximum hip height for recumbence                         |
|                              | `std_angle`                              | `40.0`           | deg          | Mode 1: Minimum torso angle for recumbence                        |
|                              | `floor_low_hip`                          | `0.28`           | scale        | Mode 2: Floor-curled maximum hip height                           |
|                              | `sofa_low_hip`                           | `0.40`           | scale        | Mode 3: Sofa/bed slump maximum hip height                         |
|                              | `chair_rejection.max_angle`              | `20.0`           | deg          | Chair sitting rejection maximum torso angle                       |
|                              | `chair_rejection.max_aspect`             | `0.70`           | aspect       | Chair sitting rejection maximum bounding box aspect               |
| **`event_lifecycle`**        | `allow_immediate_recovery`               | `true`           | bool         | Emit `fall_recovered` on fast recovery within `post_s`            |
|                              | `cooldown_s`                             | `5.0`            | s            | Minimum suppression window between identical alarms               |
|                              | `realert_s`                              | `60.0`           | s            | Escalation period for unreviewed incidents                        |
|                              | `stitch_s`                               | `3.0`            | s            | Occlusion gap threshold for track ID stitching                    |
|                              | `stitch_max_dist_ref_scale`              | `0.8`            | factor       | Maximum centroid distance for stitching ($`\times S_{\text{ref}}`$) |
| **`immobility`**             | `win_s`                                  | `10.0`           | s            | Sliding excursion window for stillness testing                    |
|                              | `disp_thresh`                            | `0.035`          | scale        | Excursion ceiling for immobility ($`\le 3.5\%`$ ref scale)        |
|                              | `T_immobile_post_fall`                   | `40.0`           | s            | Post-fall stillness duration before alert                         |
|                              | `T_immobile_lying`                       | `100.0`          | s            | General floor recumbence immobility duration                      |
|                              | `T_immobile_seated`                      | `150.0`          | s            | Chair immobility alert duration                                   |
|                              | `post_fall_recovery.min_hip`             | `0.40`           | scale        | Upright hip threshold clearing post-fall state                    |
|                              | `post_fall_recovery.active_motion_clear` | `1.20`           | scale/s      | Active motion threshold clearing post-fall state                  |

---

## 4. System Interfaces & API Integration Guide

The system exposes clean, well-typed programmatic and command-line interfaces for real-time video processing, alerting, operator review, and benchmark validation.

### 4.1 Python Streaming Engine Interface (`FallMonitor`)

Integrators can embed the state machine directly into video ingestion pipelines:

```python
from pathlib import Path
import cv2
import numpy as np
from fall_monitor import FallMonitor, load_cfg
from tsstg_model import TSSTGClassifier

# 1. Load configuration and classifier
cfg = load_cfg("config.yaml")
classifier = TSSTGClassifier(
    weight_dir=Path("weights"),
    variant="mmaction2",
    streams=("j", "jm"),
    num_classes=8,
    device="cpu"  # or "cuda:0"
)

# 2. Instantiate FallMonitor
monitor = FallMonitor(
    cfg=cfg,
    classifier=classifier,
    evidence_dir=Path("evidence_snapshots"),
    img_shape=(240, 320)
)

# 3. Stream frames in real-time
# For each video frame at timestamp t:
# ids: List[int]
# boxes: List[np.ndarray] shape (4,) in [x1, y1, x2, y2]
# kps: List[np.ndarray] shape (17, 3) in [x, y, conf]
# frame: Optional[np.ndarray] BGR image for snapshot generation
events = monitor.update(t, ids, boxes, kps, frame=frame_bgr)

for event in events:
    print(f"[{event['priority'].upper()}] {event['type']} on track {event['track']}")
```

#### Event Output Schema

When a candidate event is emitted by `monitor.update()`, it returns a structured dictionary:

```json
{
  "type": "fall",
  "track": 1,
  "t_start": 3.84,
  "t_candidate": 4.50,
  "t_alert": 4.50,
  "priority": "high",
  "status": "candidate",
  "event_id": "a1b2c3d4e5f6",
  "evidence": {
    "clip_window": [1.84, 6.50],
    "tsstg_probs": {
      "Standing": 0.05,
      "Walking": 0.02,
      "Sitting": 0.01,
      "Lying": 0.12,
      "StandUp": 0.00,
      "SitDown": 0.03,
      "Fall": 0.72,
      "Instability": 0.05
    },
    "ntu60_top5": [
      ["falling", 0.72],
      ["staggering", 0.14],
      ["lie down", 0.08],
      ["sit down", 0.04],
      ["drop", 0.02]
    ],
    "kinematics": {
      "drop": 0.22,
      "vdown": 0.28,
      "hip_rel": 0.22,
      "angle": 58.4,
      "aspect": 1.25,
      "knee_angle": 165.2,
      "disp_win": 0.015,
      "motion": 0.45
    },
    "snapshot_path": "evidence_snapshots/a1b2c3d4e5f6.jpg"
  }
}
```

#### Event Types & Priority Classification

- **`fall`** (`priority: "high"`): Sudden collapse followed by verified persistent floor recumbence.
- **`fall_recovered`** (`priority: "low"`): High-acceleration impact latch triggered, but subject successfully stood back upright within $`T_{\text{post}} \le 3.0\,\text{s}`$.
- **`lying`** (`priority: "normal"`): Controlled descent or prolonged horizontal posture without high-velocity impact lasting $`\ge T_{\text{lying}}`$.
- **`immobility`** (`priority: "normal"`): Scale-normalized joint excursion below $`\text{disp\_thresh}`$ exceeding stillness duration ($`T_{\text{immobile\_post\_fall}}`$, $`T_{\text{immobile\_lying}}`$, or $`T_{\text{immobile\_seated}}`$).

#### Operator Review API

The `FallMonitor` includes a stateful triage lifecycle for human operators or clinical review systems:

```python
# Transition an event status with audit notes
updated_record = monitor.review_event(
    event_id="a1b2c3d4e5f6",
    status="confirmed",  # "confirmed" | "dismissed" | "acknowledged"
    operator_note="Nurse dispatched to room 104; resident assisted to chair."
)
```

---

### 4.2 Standalone Command-Line Runner (`src/fall_monitor.py`)

Run real-time inference on a local video file, USB webcam, or network RTSP camera:

```powershell
# Live RTSP camera stream with live preview and event logging
python src/fall_monitor.py `
    --source "rtsp://192.168.1.100:554/stream1" `
    --config config.yaml `
    --model weights/yolo26x-pose.pt `
    --tracker bytetrack.yaml `
    --show `
    --events logs/alerts.jsonl `
    --evidence-dir logs/snapshots/

# Process recorded video file and save annotated MP4
python src/fall_monitor.py `
    --source "data/videos/le2i/Home_01/video (1).avi" `
    --config config.yaml `
    --save output_annotated.mp4 `
    --stride 2
```

#### CLI Options Reference

| Flag             | Default                   | Description                                           |
|:-----------------|:--------------------------|:------------------------------------------------------|
| `--source`       | *Required*                | Video file path, webcam index (`0`), or RTSP/HTTP URL |
| `--config`       | `config.yaml`             | Master YAML configuration file                        |
| `--model`        | `weights/yolo26x-pose.pt` | Ultralytics YOLO pose weights                         |
| `--tracker`      | `bytetrack.yaml`          | ByteTrack tracker configuration                       |
| `--stride`       | `2`                       | Frame processing stride (1 = process every frame)     |
| `--imgsz`        | `640`                     | YOLO inference image resolution                       |
| `--conf-det`     | `0.25`                    | Person detection confidence threshold                 |
| `--device`       | `None`                    | Device identifier (`cpu`, `0` for GPU)                |
| `--realert`      | `60.0`                    | Minimum seconds between repeat alerts on same track   |
| `--show`         | `False`                   | Display interactive live preview window               |
| `--save`         | `None`                    | Output path to save annotated visualization MP4       |
| `--events`       | `None`                    | File path to append alerts as JSON lines              |
| `--evidence-dir` | `None`                    | Directory path where audit JPEG snapshots are stored  |

---

### 4.3 Benchmark Evaluation Interface (`src/evaluate_benchmark.py`)

Run standardized evaluations across labeled datasets (Le2i, URFD, OmniFall) with IoU tracking and tolerance windows:

```powershell
python src/evaluate_benchmark.py `
    --manifest data/manifest.csv `
    --model weights/yolo26x-pose.pt `
    --tracker bytetrack.yaml `
    --tolerance 2.0 `
    --output-dir benchmark_results/ `
    --trace
```

#### Evaluation Outputs

- **`benchmark_results/benchmark_summary.json`**: Machine-readable metrics containing overall precision, recall, $`F_1`$, false alarms per hour (FA/h), per-video detection latencies, and look-alike activity breakdowns.
- **`benchmark_results/benchmark_summary.md`**: Publication-ready Markdown summary table comparing detected vs ground-truth intervals.

---

### 4.4 OmniFall Dataset Ingestion Interface (`src/omnifall_adapter.py`)

Translate human-annotated OmniFall labels (16-class taxonomy) into standard pipeline manifests:

```powershell
# Build evaluation manifest from OmniFall Le2i annotations
python src/omnifall_adapter.py `
    --omnifall-dir external/omnifall/ `
    --video-root data/videos/le2i/ `
    --output-dir data/ `
    --benchmark le2i
```

---

## 5. Benchmark Performance Evolution

Evaluated against the Le2i Fall Detection Dataset (190 annotated videos across 4 distinct physical environments: Coffee Room, Home, Lecture Room, and Office) and URFD (70 multi-camera sequences):

| Metric                      | Baseline (v1) | Decoupled Tuned (v2) | Conservative Clean (v3) |
|:----------------------------|:-------------:|:--------------------:|:-----------------------:|
| **True Positives (TP)**     | 87            | **109**              | 102                     |
| **False Negatives (FN)**    | 43            | **21**               | 28                      |
| **False Positives (FP)**    | 0             | 6                    | **3**                   |
| **True Negatives (TN)**     | 60            | 54                   | 57                      |
| **Recall (Sensitivity)**    | 66.92%        | **83.85%**           | 78.46%                  |
| **Precision**               | **100.0%**    | 94.78%               | 97.14%                  |
| **$`F_1`$ Score**           | 0.8018        | **0.8898**           | 0.8681                  |
| **Target Misses Recovered** | Baseline      | **22 / 34 (64.7%)**  | 15 / 34 (44.1%)         |

> [!TIP]
> Academic benchmarks feature staged falls with video capture truncated 5 to 7 seconds after impact. Fine-tuning should not be overfitted against academic datasets, but rather calibrated against domain-specific client deployment footage.

---

## 6. Demonstration Deliverables

Pre-rendered demonstration videos showcasing real-time telemetry HUDs, skeleton overlays, and clinical event badges are located in `data/demos/`:

- **`demo_fall.mp4`** (`0.66 MB`, 160 frames, H.264):
  Demonstrates rapid transition from normal walking $`\to`$ sudden kinematic collapse $`\to`$ `CRITICAL: FALL DETECTED`.
- **`demo_lying.mp4`** (`1.59 MB`, 300 frames, H.264):
  Demonstrates controlled intentional transition to the floor without high velocity $`\to`$ `MONITOR: LYING ON FLOOR` after $`T_{\text{lying}}`$ expires.
- **`demo_immobility.mp4`** (`0.94 MB`, 231 frames, H.264):
  Demonstrates fall impact at $`t = 4.24\,\text{s}`$ $`\to`$ floor recumbence at $`t = 5.80\,\text{s}`$ $`\to`$ sustained lack of excursion triggering `EMERGENCY: POST-FALL IMMOBILE`.

---

## 7. Directory Structure

```text
fall_detection_module\
├── config.yaml                     # Master decoupled configuration (sole source of truth)
├── README.md                       # Comprehensive system and interface documentation
├── requirements.txt                # Pinned dependencies
├── src\
│   ├── fall_monitor.py             # Streaming engine, state machine, vetoes, immobility & CLI
│   ├── tsstg_model.py              # ST-GCN RecognizerGCN, checkpoint loader & 8-class taxonomy
│   ├── keypoint_adapter.py         # 17-keypoint resampler, prenormalize2d & causal imputation
│   ├── evaluate_benchmark.py       # Benchmark evaluation harness, parser & IoU matching
│   ├── omnifall_adapter.py         # OmniFall annotation ingestion & manifest generator
│   └── transfer_gate.py            # Normalization validation & transfer learning hook
├── scripts\
│   ├── generate_all_demos.py       # Demonstration video rendering pipeline
│   ├── evaluate_a42_separation.py  # A42 vs A43 population AUROC evaluation
│   └── inspect_latched_cases.py    # Multi-mode posture confirmation geometry inspector
├── tests\                          # 53 automated unit & regression tests (100% passing)
│   ├── test_config_source_of_truth.py # Proves config.yaml is the sole source of truth
│   ├── test_phase1_improvements.py # Taxonomy, A42 isolation, two-tier events
│   ├── test_adversarial_critiques.py # Hardened audit regression tests
│   ├── test_fall_monitor.py        # Core state machine & lifecycle tests
│   ├── test_tsstg_model.py         # Model architecture & weight SHA256 registry
│   ├── test_keypoint_adapter.py    # Keypoint sanitization & feature generation
│   ├── test_omnifall_adapter.py    # Taxonomy & manifest mapping tests
│   └── test_transfer_gate.py       # Normalization & fine-tuning tests
├── weights\
│   ├── yolo26x-pose.pt             # YOLOv8x-pose model weights
│   ├── mmaction2_joint.pth         # MMAction2 ST-GCN joint weights
│   ├── mmaction2_jm.pth            # MMAction2 ST-GCN joint-motion weights
│   ├── mmaction2_bone.pth          # MMAction2 ST-GCN bone weights
│   ├── mmaction2_bm.pth            # MMAction2 ST-GCN bone-motion weights
│   ├── pyskl_j.pth                 # PySKL ST-GCN joint weights
│   ├── pyskl_jm.pth                # PySKL ST-GCN joint-motion weights
│   ├── pyskl_b.pth                 # PySKL ST-GCN bone weights
│   └── pyskl_bm.pth                # PySKL ST-GCN bone-motion weights
├── data\
│   ├── demos\                      # Pre-rendered demonstration MP4 videos
│   ├── manifest.csv                # Dataset evaluation manifest
│   └── videos\                     # Local benchmark video corpora (Le2i, URFD)
├── docs\phases\                    # Formal phase reports & analytical documentation
└── reports\                        # Statistical reports and JSON summaries
```

---

## 8. Quickstart & Verification Commands

```powershell
# 1. Run the entire automated test suite (53/53 tests pass)
python -m pytest tests/ -v

# 2. Run source-of-truth verification tests exclusively
python -m pytest tests/test_config_source_of_truth.py -v

# 3. Run real-time detection on a video file with annotated output
python src/fall_monitor.py `
    --source "data/videos/le2i/Home_01/video (1).avi" `
    --config config.yaml `
    --save data/demos/output_test.mp4 `
    --stride 2

# 4. Run full benchmark evaluation across dataset manifest
python src/evaluate_benchmark.py `
    --manifest data/manifest.csv `
    --output-dir benchmark_results/

# 5. Re-render all demonstration videos (Fall, Lying, Immobility)
python scripts/generate_all_demos.py --only all
```
