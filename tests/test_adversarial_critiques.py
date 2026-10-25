"""tests/test_adversarial_critiques.py - Faithful tests for critical bugs surfaced by the adversarial auditor.

Tests:
1. test_critique_1_resample_window_gap_check_isolated_to_trailing_window
2. test_critique_2_compute_windowed_displacement_missing_occluded_joints_not_zero
3. test_critique_3_post_fall_immobility_reset_on_recovery_and_movement
4. test_critique_4_diag_hist_rolling_buffer_memory_cap
5. test_critique_5_post_fall_stability_not_corrupted_by_single_frame_veto
6. test_critique_6_evaluate_benchmark_spatial_iou_matching
"""
import numpy as np
import pytest

from conftest import FakeClassifier, FALL_, LYING_, SITDOWN_, STAND_, onehot7, drive
from synth import track, hold, jitter, STAND, SIT, LIE
import fall_monitor as FM
import keypoint_adapter as KA
import evaluate_benchmark as EB


# ----------------------------------------------------------------------------------------------------------------------
# Critique 1: Gap in 30s track history must not disable ST-GCN inference in the current 1.16s window
# ----------------------------------------------------------------------------------------------------------------------
def test_critique_1_resample_window_gap_check_isolated_to_trailing_window():
    # Track spans 30s. A 1.5s tracking gap occurred early (between t=2.0 and t=3.5).
    # From t=3.5 to t=30.0, the track is continuous at 25 Hz.
    t_part1 = np.arange(0.0, 2.04, 0.04)  # 0.0 to 2.0
    t_part2 = np.arange(3.5, 30.04, 0.04) # 3.5 to 30.0 (continuous trailing 26.5s)
    t = np.concatenate([t_part1, t_part2])
    
    n_frames = len(t)
    kp = np.ones((n_frames, 17, 3), dtype=np.float32) * 100.0
    conf = np.ones((n_frames, 17), dtype=bool)

    # In the active trailing window [30.0 - 1.16, 30.0], timestamps are perfectly continuous (step 0.04s)
    res = KA.resample_window_25hz(t, kp, conf, window_s=1.16, num_frames=30, max_gap_s=1.0)
    
    # Buggy code checks np.diff(ts) on the entire 30s array and returns None due to the gap at t=2.0-3.5!
    assert res is not None, "ST-GCN was disabled at t=30.0 by a tracking gap that occurred 26 seconds ago!"
    res_pts, res_mask = res
    assert res_pts.shape == (30, 17, 3)
    assert res_mask.shape == (30, 17)


# ----------------------------------------------------------------------------------------------------------------------
# Critique 2: Occluded / missing joints must NOT be declared motionless (disp_win = 0.0)
# ----------------------------------------------------------------------------------------------------------------------
def test_critique_2_compute_windowed_displacement_missing_occluded_joints_not_zero():
    # 50 frames @ 25 Hz (2.0s window), but all joints are occluded / unconfident (mask is all False)
    t = np.arange(0.0, 2.0, 0.04)
    n = len(t)
    kp = np.zeros((n, 17, 3), dtype=np.float32)
    conf_mask = np.zeros((n, 17), dtype=bool) # zero confident joints

    disp = FM.compute_windowed_displacement(t, kp, conf_mask, ref_scale=150.0, win_s=2.0, min_valid_frames=3)
    
    # Buggy code returns 0.0, which satisfies is_still = (0.0 <= 0.035), declaring ghost immobility!
    assert np.isnan(disp) or disp > FM.CFG['immobility']['disp_thresh'], (
        f"Missing/occluded joints returned displacement {disp}, which falsely declares stillness!"
    )


# ----------------------------------------------------------------------------------------------------------------------
# Critique 3: Post-fall immobility state must reset when person recovers and moves
# ----------------------------------------------------------------------------------------------------------------------
def test_critique_3_post_fall_immobility_reset_on_recovery_and_movement():
    # Sequence:
    # 1. Person falls at t=3.0, lies still until t=5.0 (fall candidate emitted at ~3.9s)
    # 2. Person recovers! Stands up and walks with high motion from t=5.0 to t=25.0 (20s of active movement)
    # 3. Person sits at t=25.0 and sits motionless until t=55.0 (30s of sitting motionless)
    fps = 10.0
    segs = [
        hold(STAND, 3.0),
        (0.3, STAND, LIE),
        hold(LIE, 2.0),           # fall detected
        (0.5, LIE, STAND),         # recovery: stands up
        hold(STAND, 20.0),         # active upright walking
        (0.5, STAND, SIT),         # sits down
        hold(SIT, 30.0)            # seated for 30s
    ]
    t, k, b = track(segs, fps=fps)
    
    fake = FakeClassifier()
    # Classifier reports FALL during drop, then STAND, then SITTING
    def pf(ts):
        if 3.0 <= ts < 4.5:
            return onehot7(FALL_)
        elif ts >= 26.0:
            return onehot7(FM.CLASS_NAMES.index('Sitting'))
        else:
            return onehot7(STAND_)

    mon = FM.FallMonitor(FM.CFG, classifier=fake, img_shape=(720, 1280), eval_every_s=0.5)
    evs = drive(mon, fake, t, k, b, probs_fn=pf)
    
    falls = [e for e in evs if e['type'] == 'fall']
    assert len(falls) == 1, "Fall should have been detected"

    # Post-fall immobility threshold is 40.0s.
    # At t=50.0s (which is > 3.9 + 40.0s), the buggy monitor checks `t >= last_fall_cand_t + 40.0`
    # and emits a post-fall immobility alert even though the person recovered and walked for 20s!
    post_fall_imm = [
        e for e in evs 
        if e['type'] == 'immobility' and np.isclose(e['t_start'], falls[0]['t_candidate'])
    ]
    assert len(post_fall_imm) == 0, (
        "Post-fall immobility fired on a person who already recovered, walked, and sat down!"
    )


# ----------------------------------------------------------------------------------------------------------------------
# Critique 4: diag_hist rolling buffer memory cap
# ----------------------------------------------------------------------------------------------------------------------
def test_critique_4_diag_hist_rolling_buffer_memory_cap():
    # In a long CCTV stream (e.g. 3000 frames @ 25 Hz), diag_hist must not grow without bound
    hist = []
    box = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)
    for _ in range(3000):
        FM.update_ref_scale(hist, box)
    
    # Should be bounded to a rolling window (e.g. <= 1500 entries = 60s @ 25fps)
    assert len(hist) <= 1500, f"diag_hist grew unbounded to {len(hist)} entries (memory leak)!"


# ----------------------------------------------------------------------------------------------------------------------
# Critique 5: Post-fall stability not corrupted by single-frame veto
# ----------------------------------------------------------------------------------------------------------------------
def test_critique_5_post_fall_stability_not_corrupted_by_single_frame_veto():
    # Person falls and lies still for 2.0s.
    # If a controlled veto occurs for 1 anomalous frame at t=4.0, it must not retroactively
    # wipe out the entire historical stability window if the rest of the window is stable lying.
    t, k, b = track([hold(STAND, 3.0), (0.3, STAND, LIE), hold(LIE, 3.0)], fps=25.0)
    fake = FakeClassifier()
    # At t=3.8s, classifier gives a 1-frame anomaly (says Sit Down), but otherwise says Fall Down then Lying
    def pf(ts):
        if 3.80 <= ts <= 3.84:
            return onehot7(SITDOWN_)
        elif 3.0 <= ts < 3.8:
            return onehot7(FALL_)
        else:
            return onehot7(LYING_)
            
    mon = FM.FallMonitor(FM.CFG, classifier=fake, img_shape=(720, 1280), eval_every_s=0.1)
    evs = drive(mon, fake, t, k, b, probs_fn=pf)
    falls = [e for e in evs if e['type'] == 'fall']
    assert len(falls) == 1, "Fall should still be detected despite a 1-frame classifier glitch during stability"


# ----------------------------------------------------------------------------------------------------------------------
# Critique 6: Benchmark evaluation requires spatial overlap (IoU >= 0.3) when GT boxes exist
# ----------------------------------------------------------------------------------------------------------------------
def test_critique_6_evaluate_benchmark_spatial_iou_matching(tmp_path):
    # GT annotation: Fall at t=0.4s to 0.8s on Person 1 at box [100, 100, 200, 300]
    ann = tmp_path / "video1.txt"
    ann.write_text("10\n20\n10,1,100,100,200,300\n11,1,100,100,200,300\n")
    
    # Simulated video run record where an alert fired at t=0.5s, but on Person 2 at [800, 800, 900, 900]
    # Spatial IoU with Person 1 is 0.0!
    detected_box = np.array([800.0, 800.0, 900.0, 900.0])
    gt_box = np.array([100.0, 100.0, 200.0, 300.0])
    
    iou = EB.compute_box_iou(detected_box, gt_box)
    assert iou == 0.0
    
    # If match_fall_event requires min_iou=0.3, this alert should be classified as a False Alarm, NOT a TP!
    gt = EB.parse_le2i_annotation(ann, fps=25.0)
    alert = dict(type='fall', t_start=0.5, t_candidate=0.6, t_alert=0.6, box=detected_box)
    
    is_tp = EB.match_fall_event_to_gt(alert, gt, tolerance_s=2.0, min_iou=0.3)
    assert is_tp is False, "Spatial mismatch (different person) was falsely credited as a True Positive!"

# -----------------------------------------------------------------------
# Critique 7: Path C geometry conditions are mutually exclusive with
# controlled_bend geometry — verifies design correctness of cold-start path.
# -----------------------------------------------------------------------
def test_critique_7_path_c_geometry_precludes_controlled_bend():
    """
    Path C requires: angle > 55 OR aspect > 0.90 OR head < 0.
    controlled_bend requires: angle < 45 AND aspect < 0.85 AND head > 0.30.
    These are geometrically mutually exclusive, so a controlled bend cannot
    satisfy Path C's trigger condition — no veto is needed in Path C itself.

    (a) A genuine cold-start fall (person enters frame already horizontal) MUST
        be detected — confirming Path C works correctly for real emergencies.
    (b) A cold-start controlled sit-down (upright geometry) must NOT trigger
        because it fails Path C condition 5 (angle/aspect/head) naturally.
    """
    fps = 25.0

    # (a) True cold-start fall: person detected already horizontal.
    segs_fall = [
        hold(STAND, 0.2),
        (0.1, STAND, LIE),
        hold(LIE, 2.0),
    ]
    t_fall, k_fall, b_fall = track(segs_fall, fps=fps)
    mon_fall = FM.FallMonitor(FM.CFG, classifier=None, img_shape=(720, 1280), eval_every_s=0.08)
    evs_fall = drive(mon_fall, None, t_fall, k_fall, b_fall, probs_fn=None)
    assert any(e["type"] == "fall" for e in evs_fall), (
        "Path C failed to detect a genuine cold-start fall (person entered frame already fallen)!"
    )

    # (b) Upright-to-seated transition: angle stays < 45, aspect stays < 0.85.
    #     Path C condition 5 is never satisfied => no false alert.
    segs_sit = [
        hold(STAND, 0.3),
        (0.6, STAND, SIT),
        hold(SIT, 2.0),
    ]
    t_sit, k_sit, b_sit = track(segs_sit, fps=fps)
    mon_sit = FM.FallMonitor(FM.CFG, classifier=None, img_shape=(720, 1280), eval_every_s=0.08)
    evs_sit = drive(mon_sit, None, t_sit, k_sit, b_sit, probs_fn=None)
    falls_sit = [e for e in evs_sit if e["type"] == "fall"]
    assert len(falls_sit) == 0, (
        f"Path C triggered on a controlled sit-down — false positive! {falls_sit}"
    )
