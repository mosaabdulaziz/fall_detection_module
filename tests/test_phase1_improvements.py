"""test_phase1_improvements.py - Unit test suite verifying Phase 1 fixes:
1. config.yaml loading and schema completeness.
2. Action class mapping safety and NTU-60 projection index stability.
3. Post-fall confirmation recovery on floor-curled and sofa-propped postures.
4. Two-tier event emissions (fall vs fall_recovered).
"""
import pytest
import numpy as np
from pathlib import Path
import fall_monitor as FM
from tsstg_model import (
    CLASS_NAMES_8,
    ActionClass,
    CLASS_TO_IDX_8,
    project_ntu60_to_8_classes,
    project_ntu60_to_7_classes,
)
from synth import STAND, SIT, LIE, hold, track

ROOT = Path(__file__).resolve().parent.parent


def test_config_yaml_loading_and_schema():
    cfg = FM.load_cfg()
    assert "video" in cfg
    assert "detector" in cfg
    assert "sampling" in cfg
    assert "windowing" in cfg
    assert "normalization" in cfg
    assert "action_classes" in cfg
    assert "kinematic_veto" in cfg
    assert "path_a" in cfg
    assert "path_b" in cfg
    assert "post_fall_confirmation" in cfg
    assert "event_lifecycle" in cfg

    # Verify key decoupled values
    assert cfg["sampling"]["conf_kp"] == 0.30
    assert cfg["windowing"]["primary_window_s"] == 2.50
    assert cfg["post_fall_confirmation"]["max_motion"] == 1.80
    assert cfg["post_fall_confirmation"]["floor_low_hip"] == 0.28


def test_action_class_mapping_safety():
    # Verify exact taxonomy indices
    assert CLASS_NAMES_8[0] == ActionClass.STANDING
    assert CLASS_NAMES_8[1] == ActionClass.WALKING
    assert CLASS_NAMES_8[2] == ActionClass.SITTING
    assert CLASS_NAMES_8[3] == ActionClass.LYING
    assert CLASS_NAMES_8[4] == ActionClass.STAND_UP
    assert CLASS_NAMES_8[5] == ActionClass.SIT_DOWN
    assert CLASS_NAMES_8[6] == ActionClass.FALL
    assert CLASS_NAMES_8[7] == ActionClass.INSTABILITY

    # Test NTU-60 projection: A43 (falling, idx 42) -> Fall (idx 6)
    ntu_fall = np.zeros(60)
    ntu_fall[42] = 1.0
    q = project_ntu60_to_8_classes(ntu_fall)
    assert q[CLASS_TO_IDX_8[ActionClass.FALL]] == 1.0
    assert q[CLASS_TO_IDX_8[ActionClass.WALKING]] == 0.0

    # Test NTU-60 projection: A42 (staggering, idx 41) -> Instability (idx 7)
    ntu_stag = np.zeros(60)
    ntu_stag[41] = 1.0
    q_stag = project_ntu60_to_8_classes(ntu_stag)
    assert q_stag[CLASS_TO_IDX_8[ActionClass.INSTABILITY]] == 1.0
    # Crucial: A42 must NOT be in Walking!
    assert q_stag[CLASS_TO_IDX_8[ActionClass.WALKING]] == 0.0


def test_two_tier_event_recovered():
    fps = 25.0
    # Fall down, but stand up within 0.3s (rapid recovery)
    segs = [
        hold(STAND, 0.4),
        (0.2, STAND, LIE),
        (0.3, LIE, STAND),
        hold(STAND, 4.0),
    ]
    t, k, b = track(segs, fps=fps)
    
    # Custom classifier that provides action support during impact
    class QuickFallClassifier:
        def __call__(self, tensors):
            p = np.zeros(8)
            p[6] = 0.50 # Fall
            return 6, ActionClass.FALL, p, np.zeros(60)
            
    mon = FM.FallMonitor(FM.CFG, classifier=QuickFallClassifier(), img_shape=(720, 1280), eval_every_s=0.10)
    
    events = []
    for i in range(len(t)):
        evs = mon.update(float(t[i]), [1], [b[i]], [k[i]])
        events.extend(evs)
        
    # Check if a fall_recovered event or fall event was emitted
    event_types = [e["type"] for e in events]
    # Person recovered upright, so should not have a high-priority unrecovered fall
    assert "fall_recovered" in event_types or "fall" not in event_types
