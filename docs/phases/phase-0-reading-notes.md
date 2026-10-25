# Phase 0 — Reading Notes (rewritten to match `research/plan.md` + `design-3.md`)

**Date**: 2026-10-22 · supersedes the earlier draft, which described modules/thresholds that are not in the plan.

## Sources read
| File | Used for |
|---|---|
| `research/problem.md` | 3 candidate-event types, operator review, JSONL + annotated video output, Le2i evaluation |
| `research/design/design-3.md` | Stage 1–4 design, departures `UNTESTED DEPARTURE 1–5`, paper/code divergences |
| `research/plan.md` | 5 modules, interfaces, test matrix T1–T16 |
| `research/ledger/evidence.jsonl` (E201–E205), `research/sources/S2.md` | Upstream MMAction2/PYSKL facts |
| `mmaction2_src_ref/` | Upstream source: `graph.py`, `gcn_utils.py`, `stgcn_backbone.py`, `gcn_head.py`, `recognizer_gcn.py`, `pose_transforms.py`, `formatting.py`, `label_map_ntu60.txt` |
| `vendor/s0_fall_monitor.py` (copy of the baseline `fall_monitor.py`) | `hold_runs`, `merge_runs`, `compute_features`, `detect_falls`, `detect_lying`, `FallMonitor`, `run`, CLI |

## Architecture (per plan)
`yolo26x-pose.pt` + ByteTrack → `keypoint_adapter.py` (sanitise 17 kpts, 25 Hz / 30-frame resample, `PreNormalize2D → GenSkeFeat → UniformSampleFrames(100) → FormatGCNInput(M=2)`) → `tsstg_model.py` (10-stage ST-GCN, 60 NTU classes → 7 classes) → `fall_monitor.py` (smoothing, veto, pending-fall latch, lying, windowed-excursion immobility, stitching, lifecycle/evidence) → `evaluate_benchmark.py`. `transfer_gate.py` checks YOLO26→NTU-60 transfer and holds the fine-tune hook.

## Reuse policy (user instruction, 2026-10-22)
Use what already exists; write only the glue the docs require.
| Need | Reuse |
|---|---|
| Graph / `unit_gcn` / `unit_tcn` / `STGCN` / `GCNHead` | Upstream MMAction2 (`mmaction2_src_ref/`; `graph.py` has no mmcv dependency). Phase-3 import probe decides: (1) `pip` `mmengine` + `mmcv-lite` + `mmaction2 --no-deps` (resolver dry-run succeeded for mmaction2 1.2.0 / mmcv-lite 2.2.0 / mmengine 0.10.7), else (2) vendored upstream files plus a minimal registry/`build_norm_layer` shim. Hand-written reimplementation only with user approval. |
| Pose transforms (`PreNormalize2D`, `GenSkeFeat`, `UniformSampleFrames`, `FormatGCNInput`) | Upstream transforms; adapter only wraps them |
| Detection + tracking | `ultralytics` `model.track(..., tracker='bytetrack.yaml')` |
| Interpolation, percentile, median | `numpy` / `pandas` / `scipy` |
| `hold_runs`, `merge_runs`, `compute_features`, `detect_falls`, `detect_lying`, `FallMonitor`, `run`, CLI | The baseline (vendored) — extended, not rewritten |
| Checkpoint SHA-256, loading | `hashlib`, `torch.load(weights_only=True)` |

## Environment (done)
`F:\fall_detection\.venv` (Python 3.14, torch 2.14 cu126, ultralytics 8.4.183, lapx, pytest); 8 OpenMMLab checkpoints SHA-256 verified; `yolo26x-pose.pt` cached in `weights/`. All caches under `F:\fall_detection`.
