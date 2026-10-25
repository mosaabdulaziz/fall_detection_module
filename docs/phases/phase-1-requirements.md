# Phase 1 — Requirements (rewritten to match `research/plan.md` + `design-3.md`)

**Paths**: the plan's `C:\...\scratch\fall_detection_research\` maps to `F:\fall_detection\` (flat modules at the project root, tests in `tests/`).

## Modules and contracts
| Module | Contract |
|---|---|
| `tsstg_model.py` | `Graph`, `normalize_digraph`, `RecognizerGCN()` (callable on `(N,2,100,17,3)` → logits `(N,60)`; `.backbone.gcn` (10 blocks), `.backbone.data_bn`, `.cls_head.fc`), `WEIGHT_SHA256_REGISTRY`, `verify_sha256(path, expected=None)` (raises `ValueError` on mismatch), `load_recognizer_gcn_checkpoint(model, path, device, expected_sha256=None)` (maps `fc_cls→fc`, `strict=True`, `weights_only=True`, checks `A` first), `project_ntu60_to_7_classes`, `CLASS_NAMES`, `NTU60_CLASS_NAMES`, `TSSTGClassifier(weight_dir, variant, streams, stream_weights, device)` — **callable**: `clf(stream_tensors) -> (pred_idx_7, name_7, probs_7 (7,), ntu60_probs (60,))` |
| `keypoint_adapter.py` | `COCO_BONE_PAIRS`, `sanitize_keypoints_17`, `resample_window_25hz` (returns `None` if <6 obs, span <1.16 s, or a step >1.0 s), `prenormalize_2d`, `gen_ske_feat(j (T,17,3), conf_mask, streams) -> {name: (1,T,17,3)}`, `uniform_sample_and_format_gcn`, `build_mmaction2_tensors` → `{stream: (1,2,100,17,3) float32}` |
| `transfer_gate.py` | `evaluate_tsstg_transfer_on_sequences(sequences, classifier, norm_modes)` with sequences `{'label','pts' (30,17,3 px),'conf_mask','img_shape'}`, labels `Fall Down / Sit down / Pickup / Standing / Walking`; returns per-mode `fall_recall, sitting_specificity, pickup_specificity, macro_f1` + `recommended_norm_mode`. `finetune_tsstg_on_yolo26(...)` → `{'loss_start','loss_end'}`, writes `out_weight_path`, keeps `A` frozen |
| `fall_monitor.py` | Baseline API preserved (`CFG`, `hold_runs`, `merge_runs`, `detect_lying`, `FallMonitor.update(t, ids, boxes, kps, frame=None)`, `run(a)`, CLI flags incl. new `--evidence-dir`). `update_ref_scale(diag_history, box, max_growth=1.15, ref_scale_init=None)` (appends in place), `compute_windowed_displacement`, `compute_knee_angle`, `FallMonitor.review_event(event_id, status, note='')` |
| `evaluate_benchmark.py` | `parse_le2i_annotation`, `compute_benchmark_metrics`, `evaluate_video_manifest` |

## Event schema (JSONL / returned dicts)
`type` (`fall|lying|immobility`), `track`, `t_start`, `t_candidate`, `t_alert`, `priority`, `event_id`, `status` (`candidate`→`acknowledged|confirmed|dismissed`), `evidence` {`clip_window`, `tsstg_probs`, `ntu60_top5`, `kinematics`, `snapshot_path`}.
Immobility durations are measured from the event's `t_start`: post-fall = fall `t_candidate` (≥5 s), lying episode start (≥10 s), seated episode start (≥120 s).

## Acceptance tests
T1–T16 exactly as in `research/plan.md` §"Test suite with source mapping"; concrete assertions and the deviations listed in `phase-2-tests.md`.

## Constraints
`STORAGE-1` (everything on `F:\fall_detection`), checkpoints verified by SHA-256 and loaded with `weights_only=True`, reuse-first policy (see Phase 0), no hand-reimplementation of anything that upstream/baseline already provides.
