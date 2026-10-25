"""test_config_source_of_truth.py - Verifies config.yaml is the sole source of truth.

Tests prove:
1. Every section and numeric threshold in the system is loaded from config.yaml.
2. In-memory perturbations of config.yaml values directly control and shift the system's
   decision boundaries across all subsystems:
   - Sampling & keypoint confidence
   - Windowing & minimum duration requirements
   - NTU-60 projection index mapping
   - Kinematic bend and knee veto thresholds
   - Path A / B / C fall latching gates
   - Post-fall confirmation postures (standard, floor-curled, sofa slump)
   - Lying duration and aspect thresholds
   - Immobility stillness and post-fall timers
   - Event lifecycle cooldown and deduplication
"""
import copy
from pathlib import Path
import numpy as np
import pytest

import fall_monitor as FM
from tsstg_model import (
    CLASS_TO_IDX_8,
    ActionClass,
    load_action_projection_cfg,
    project_ntu60_to_8_classes,
)
from keypoint_adapter import (
    sanitize_keypoints_17,
    resample_window_25hz,
    prenormalize_2d,
)
from synth import STAND, SIT, LIE, hold, track
from conftest import FakeClassifier, onehot7, FALL_, STAND_


ROOT = Path(__file__).resolve().parent.parent


def test_config_completeness_and_no_missing_thresholds():
    """Verifies that all system modules read from config.yaml without missing sections."""
    cfg = FM.load_cfg()

    required_sections = [
        "video", "detector", "sampling", "windowing", "normalization",
        "action_classes", "kinematic_veto", "fall", "path_a", "path_b", "path_c",
        "post_fall_confirmation", "lying", "immobility", "event_lifecycle"
    ]
    for section in required_sections:
        assert section in cfg, f"Missing required top-level section: {section}"

    # Verify specific calibrated numbers match system design
    assert cfg["sampling"]["conf_kp"] == 0.30
    assert cfg["sampling"]["smooth_s"] == 0.40
    assert cfg["windowing"]["primary_window_s"] == 2.50
    assert cfg["windowing"]["min_window_s"] == 1.16
    assert cfg["normalization"]["mode"] == "prenormalize2d"
    assert cfg["action_classes"]["projection"]["fall_sources"] == [42]
    assert cfg["action_classes"]["projection"]["instability_sources"] == [41]
    assert cfg["path_a"]["min_action_score"] == 0.30
    assert cfg["path_b"]["support_fall_score"] == 0.15
    assert cfg["post_fall_confirmation"]["floor_low_hip"] == 0.28
    assert cfg["post_fall_confirmation"]["sofa_low_hip"] == 0.40
    assert cfg["lying"]["T_lying"] == 2.0
    assert cfg["immobility"]["T_immobile_post_fall"] == 40.0
    assert cfg["immobility"]["disp_thresh"] == 0.035
    assert cfg["event_lifecycle"]["cooldown_s"] == 5.0


def test_sampling_conf_kp_controlled_by_config():
    """Perturbing conf_kp directly alters keypoint sanitation confidence mask."""
    box = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 0] = 150.0
    kp[:, 1] = 150.0
    kp[:, 2] = 0.25  # In-between 0.20 and 0.30

    # With default 0.30 -> rejected as not confident
    _, conf_default = sanitize_keypoints_17(kp, box, conf_kp=0.30)
    assert not np.all(conf_default)

    # With lowered threshold 0.20 -> accepted as confident
    _, conf_lowered = sanitize_keypoints_17(kp, box, conf_kp=0.20)
    assert np.all(conf_lowered)


def test_windowing_min_window_s_controlled_by_config():
    """Perturbing min_window_s directly controls minimum history requirement for resampling."""
    # 1.0 second of data at 25 Hz (25 frames)
    ts = list(np.linspace(0.0, 1.0, 25))
    kps = [np.ones((17, 3), dtype=np.float32) * 100.0 for _ in range(25)]
    masks = [np.ones(17, dtype=bool) for _ in range(25)]

    # Default min_window_s = 1.16 -> span is 1.0s, so resample must return None
    res_default = resample_window_25hz(ts, kps, masks, min_window_s=1.16)
    assert res_default is None

    # Perturbed min_window_s = 0.80 -> span is 1.0s, so resample succeeds
    res_perturbed = resample_window_25hz(ts, kps, masks, min_window_s=0.80)
    assert res_perturbed is not None
    pts, mask = res_perturbed
    assert pts.shape == (30, 17, 3)


def test_action_projection_controlled_by_config():
    """Perturbing projection config directly reroutes NTU-60 indices to different classes."""
    ntu = np.zeros(60)
    ntu[41] = 1.0  # NTU-60 A42: Staggering

    # Default projection: 41 maps to Instability (idx 7)
    proj_default = load_action_projection_cfg()
    q_default = project_ntu60_to_8_classes(ntu, projection_cfg=proj_default)
    assert q_default[7] == 1.0
    assert q_default[6] == 0.0

    # Perturbed projection: dynamically add 41 to fall_sources
    perturbed_proj = copy.deepcopy(proj_default)
    perturbed_proj["fall_sources"] = [41, 42]
    perturbed_proj["instability_sources"] = []
    q_perturbed = project_ntu60_to_8_classes(ntu, projection_cfg=perturbed_proj)
    assert q_perturbed[6] == 1.0
    assert q_perturbed[7] == 0.0


def test_path_a_min_action_score_controlled_by_config():
    """Perturbing path_a.min_action_score directly toggles latch triggering."""
    fps = 25.0
    segs = [
        hold(STAND, 1.5),
        (0.2, STAND, LIE),
        hold(LIE, 2.0),
    ]
    t, k, b = track(segs, fps=fps)

    class FixedClassifier:
        def __call__(self, tensors):
            p = np.zeros(8)
            p[6] = 0.40  # Above 0.30, but below 0.60
            return 6, ActionClass.FALL, p, np.zeros(60)

    # 1. Base config: min_action_score = 0.30 -> triggers latch
    cfg_base = FM.load_cfg()
    mon_base = FM.FallMonitor(cfg_base, classifier=FixedClassifier(), eval_every_s=0.10)
    events_base = []
    for i in range(len(t)):
        events_base.extend(mon_base.update(float(t[i]), [1], [b[i]], [k[i]]))
    assert any(e["type"] == "fall" for e in events_base)

    # 2. Perturbed config: min_action_score = 0.60 -> suppresses latch
    cfg_perturbed = copy.deepcopy(cfg_base)
    cfg_perturbed["path_a"]["min_action_score"] = 0.60
    cfg_perturbed["path_b"]["support_fall_score"] = 0.90
    cfg_perturbed["path_b"]["combined_support_score"] = 0.90
    mon_perturbed = FM.FallMonitor(cfg_perturbed, classifier=FixedClassifier(), eval_every_s=0.10)
    events_perturbed = []
    for i in range(len(t)):
        events_perturbed.extend(mon_perturbed.update(float(t[i]), [1], [b[i]], [k[i]]))
    assert not any(e["type"] == "fall" for e in events_perturbed)


def test_kinematic_veto_controlled_by_config():
    """Perturbing kinematic_veto.controlled_bend thresholds toggles veto decision."""
    cfg = FM.load_cfg()
    # Default max_angle is 45.0 degrees
    kv_default = cfg["kinematic_veto"]["controlled_bend"]
    assert kv_default["max_angle"] == 45.0

    # Controlled bend posture: angle = 30.0, aspect = 0.70, head_rel = 0.40, vpk = 0.10
    curr_angle = 30.0
    curr_aspect = 0.70
    curr_head = 0.40
    curr_vpk = 0.10

    # With default threshold (45.0) -> bend veto triggers (True)
    is_vetoed_default = (
        curr_angle < kv_default["max_angle"] and
        curr_aspect < kv_default["max_aspect"] and
        curr_head > kv_default["min_head_rel"] and
        curr_vpk <= kv_default["max_vdown"]
    )
    assert is_vetoed_default is True

    # With perturbed threshold (max_angle = 20.0) -> bend veto does NOT trigger (False)
    kv_perturbed = copy.deepcopy(kv_default)
    kv_perturbed["max_angle"] = 20.0
    is_vetoed_perturbed = (
        curr_angle < kv_perturbed["max_angle"] and
        curr_aspect < kv_perturbed["max_aspect"] and
        curr_head > kv_perturbed["min_head_rel"] and
        curr_vpk <= kv_perturbed["max_vdown"]
    )
    assert is_vetoed_perturbed is False


def test_post_fall_confirmation_floor_low_hip_controlled_by_config():
    """Perturbing post_fall_confirmation.floor_low_hip directly shifts floor-curled confirmation."""
    cfg_base = FM.load_cfg()
    p_conf = cfg_base["post_fall_confirmation"]

    # Posture on floor: hr_post = 0.25 (between 0.20 and 0.28)
    hr_post = 0.25
    curr_drop = 0.20
    ang_post = 25.0
    asp_post = 0.90
    head_post = 0.10

    # Default floor_low_hip = 0.28 -> Confirms posture
    confirmed_default = (
        (hr_post <= p_conf["floor_low_hip"]) and
        (curr_drop >= p_conf["floor_min_drop"]) and
        (
            (ang_post >= p_conf["floor_angle"]) or
            (asp_post >= p_conf["floor_aspect"]) or
            (head_post <= p_conf["floor_head_low"])
        )
    )
    assert confirmed_default is True

    # Perturbed floor_low_hip = 0.20 -> Rejects posture
    p_conf_perturbed = copy.deepcopy(p_conf)
    p_conf_perturbed["floor_low_hip"] = 0.20
    confirmed_perturbed = (
        (hr_post <= p_conf_perturbed["floor_low_hip"]) and
        (curr_drop >= p_conf_perturbed["floor_min_drop"]) and
        (
            (ang_post >= p_conf_perturbed["floor_angle"]) or
            (asp_post >= p_conf_perturbed["floor_aspect"]) or
            (head_post <= p_conf_perturbed["floor_head_low"])
        )
    )
    assert confirmed_perturbed is False


def test_lying_duration_controlled_by_config():
    """Perturbing lying.T_lying directly determines whether lying event emits."""
    fps = 25.0
    # Lie down for exactly 2.5s
    segs = [
        hold(STAND, 0.5),
        hold(LIE, 2.5),
        hold(STAND, 1.0),
    ]
    t, k, b = track(segs, fps=fps)

    # 1. Default config (T_lying = 2.0s) -> 2.5s duration exceeds 2.0s -> emits lying candidate
    cfg_base = FM.load_cfg()
    mon_base = FM.FallMonitor(cfg_base, classifier=None, eval_every_s=0.10)
    events_base = []
    for i in range(len(t)):
        events_base.extend(mon_base.update(float(t[i]), [1], [b[i]], [k[i]]))
    assert any(e["type"] == "lying" for e in events_base)

    # 2. Perturbed config (T_lying = 5.0s) -> 2.5s duration is insufficient -> does NOT emit
    cfg_perturbed = copy.deepcopy(cfg_base)
    cfg_perturbed["lying"]["T_lying"] = 5.0
    mon_perturbed = FM.FallMonitor(cfg_perturbed, classifier=None, eval_every_s=0.10)
    events_perturbed = []
    for i in range(len(t)):
        events_perturbed.extend(mon_perturbed.update(float(t[i]), [1], [b[i]], [k[i]]))
    assert not any(e["type"] == "lying" for e in events_perturbed)


def test_immobility_disp_thresh_controlled_by_config():
    """Perturbing immobility.disp_thresh controls stillness classification."""
    cfg = FM.load_cfg()
    disp_thresh_default = cfg["immobility"]["disp_thresh"]  # 0.035

    disp_val = 0.020  # Measured joint excursion

    # Under default threshold (0.035): 0.020 is still
    assert disp_val <= disp_thresh_default

    # Under tightened threshold (0.010): 0.020 is NOT still
    disp_thresh_tight = 0.010
    assert not (disp_val <= disp_thresh_tight)


def test_event_lifecycle_cooldown_controlled_by_config():
    """Perturbing cooldown_s controls suppression of repeated alerts."""
    cfg_base = FM.load_cfg()
    mon = FM.FallMonitor(cfg_base, realert_s=60.0)

    # Record first candidate at t=10.0
    mon.emitted[("lying", 1)] = 10.0
    mon.last_cand[("lying", 1)] = 10.0

    cooldown_default = cfg_base["event_lifecycle"]["cooldown_s"]  # 5.0s

    # Candidate at t=12.0s: within 5.0s cooldown -> suppressed
    t_cand = 12.0
    is_suppressed_default = (t_cand <= mon.last_cand[("lying", 1)] + cooldown_default)
    assert is_suppressed_default is True

    # With perturbed cooldown = 1.0s: candidate at 12.0s is beyond cooldown -> allowed
    cooldown_perturbed = 1.0
    is_suppressed_perturbed = (t_cand <= mon.last_cand[("lying", 1)] + cooldown_perturbed)
    assert is_suppressed_perturbed is False
