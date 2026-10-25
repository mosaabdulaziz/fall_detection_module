# Phase 2 — Tests (TDD red)

**Location**: `F:\fall_detection\tests\` (`conftest.py`, `synth.py`, 5 test files) + `pytest.ini`. Run: `.venv\Scripts\python.exe -m pytest`.
**Red status**: all 5 modules fail at collection with `ModuleNotFoundError` for `tsstg_model`, `keypoint_adapter`, `transfer_gate`, `fall_monitor`, `evaluate_benchmark` — the expected reason. Syntax of every test file compiles.

## Map to the plan
| Plan | Test(s) |
|---|---|
| T1 | `test_mmaction2_stgcn_ten_stages_matches_code_divergence` |
| T2 | `test_coco_17_graph_matches_checkpoint_buffer_before_load` |
| T3 | `test_mmaction2_and_pyskl_checkpoints_load_strictly_with_sha256` (×5), `test_tampered_checkpoint_is_rejected`, `test_projection_ntu60_to_7_classes_is_a_distribution` |
| T4 | `test_mmaction2_tsstg_discriminates_...` (`real_weights`) |
| T5–T8 | `test_keypoint_adapter.py` (same names as plan) |
| T9 | `test_transfer_gate_evaluates_norm_modes_and_finetune_step`, `test_transfer_gate_real_weights_fall_and_sit` (`real_weights`) |
| T10–T15 | `test_fall_monitor.py` (+ `test_compute_knee_angle_...`, `test_floor_work_and_walking_never_emit_candidates`, `test_immobility_durations_post_fall_lying_and_seated` as T14 extensions) |
| T16 | `test_le2i_annotation_parser_and_heldout_benchmark_metrics` |

## Fixture validation
`tests/synth.py` synthetic skeletons were run through the unmodified baseline (`vendor/s0_fall_monitor.py`): the fall track emits `fall`, the sit-down track emits nothing, static lying emits `lying`, and the 27-frame cold-start track emits `fall`.

## Verifier report (independent agent) — dispositions
Fixed: weak T10 equality, vacuous `or True`, T5 constant scores, T12(c) loose bounds, T14 mirrored/stale-motion cases, T9 real-weights pickup, T15 priority/`t_alert`/kinematics/`clip_window`/top-5, added floor-work, walking, knee-angle tests.
Deviations from the plan, all deliberate:
- T7: plan says jm non-zero in all 100 frames; the last source frame is zero by definition (`ToMotion`), so the test allows exactly those frames.
- T10: plan signature lacks `ref_scale_init` and in-place history append; added to Phase-1 contract.
- T11(b): plan says 20 frames; fixture uses 27 frames (span 1.04 s < 1.16 s), which is the real cold-start criterion.
- T14 durations measured from event `t_start` (defined in Phase 1).
Open (not yet covered — needs your call):
1. `evaluate_video_manifest` is only existence-checked; the manifest schema is not defined in the plan.
2. Knee-angle/crouch veto (`angle<45, aspect<0.85, head_rel>0.30, vpk<=0.30`) has only the `compute_knee_angle` unit test; no end-to-end discriminating sequence yet.
3. Not tested: `track_compact` numerics, two simultaneous tracks, `acknowledged`/`dismissed` transitions, realert expiry.
4. T4 / real-weights T9 use thresholds from the plan on procedural stick figures (risk: may be unachievable; would then need real skeleton clips from `data/videos/`).
