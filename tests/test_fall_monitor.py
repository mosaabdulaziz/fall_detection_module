"""T10-T15: fall_monitor.py - hybrid fall / lying / immobility monitor (plan Task 4).

The ST-GCN is replaced by FakeClassifier (conftest) so the state machine is tested in isolation.
"""
import argparse
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import fall_monitor as FM  # ModuleNotFoundError == expected red
from conftest import FakeClassifier, drive, onehot7, STAND_, SITTING_, LYING_, SITDOWN_, FALL_
from synth import STAND, SIT, LIE, hold, track, jitter, IMG_SHAPE

H_, W_ = IMG_SHAPE


def _mon(fake, **kw):
    kw.setdefault("eval_every_s", 0.0)
    return FM.FallMonitor(FM.CFG, classifier=fake, img_shape=IMG_SHAPE, **kw)


def _falls(evs):
    return [e for e in evs if e["type"] == "fall"]


FALL_SEGS = [hold(STAND, 3.0), (0.3, STAND, LIE), hold(LIE, 5.0)]


# ---------------------------------------------------------------- T10
def test_ref_scale_outlier_clamped_and_non_decaying():
    stand, lying = np.array([0, 0, 50, 180.0]), np.array([0, 0, 180, 50.0])
    assert np.isclose(FM.update_ref_scale([], stand), FM.update_ref_scale([], lying))
    d0 = float(np.hypot(50, 180))

    hist = []
    for _ in range(600):                                        # 24 s of compressed sitting @ 25 Hz
        ref = FM.update_ref_scale(hist, np.array([0, 0, 50, 90.0]), ref_scale_init=d0)
    assert ref >= d0 - 1e-6                                      # never decays below the standing reference

    hist = [d0] * 8
    FM.update_ref_scale(hist, np.array([0, 0, 300, 220.0]))
    ref = FM.update_ref_scale(hist, np.array([0, 0, 300, 220.0]))   # 2-frame two-person bbox merge spike
    med = float(np.median(hist))
    assert ref <= 1.15 * med + 1e-6 and ref > 1.10 * med          # clamp is binding (raw p90 would be the 2x spike)


# ---------------------------------------------------------------- T11
def test_track_stitching_and_cold_start_under_30_frames():
    # (a) ID switch mid-fall: tid 1 -> tid 2 right after the drop, person already lying nearby
    t, k, b = track(FALL_SEGS, fps=25.0)
    fake = FakeClassifier(onehot7(FALL_))
    mon = _mon(fake)
    evs = []
    for i in range(len(t)):
        tid = 1 if t[i] < 3.4 else 2
        evs += mon.update(float(t[i]), [tid], [b[i]], [k[i]])
    falls = _falls(evs)
    assert len(falls) == 1 and falls[0]["track"] == 2 and falls[0]["priority"] == "high"

    # (b) < 6 frames of history / span < 1.16 s: kinematic cold-start path, classifier never consulted
    t, k, b = track([hold(STAND, 0.24), (0.08, STAND, LIE), hold(LIE, 0.76)], fps=25.0)
    assert t[-1] - t[0] < 1.16
    fake3 = FakeClassifier(onehot7(STAND_)); mon3 = _mon(fake3)
    falls = _falls(drive(mon3, fake3, t, k, b))
    assert len(falls) == 1 and fake3.calls == 0


# ---------------------------------------------------------------- T12
def test_hybrid_fall_latch_requires_descent_and_lying_stability_and_vetoes_seated():
    # (a) sits down, then the classifier insists "Fall Down" for 2.4 s: seated posture can never be lying-stable
    t, k, b = track([hold(STAND, 1.0), (0.5, STAND, SIT), hold(SIT, 5.0)], fps=25.0)
    fake = FakeClassifier()
    pf = lambda ts: onehot7(FALL_) if 3.0 <= ts < 5.4 else onehot7(SITTING_)
    assert _falls(drive(_mon(fake), fake, t, k, b, probs_fn=pf)) == []

    # (b) identical collapse kinematics, classifier says "Sit down": vetoed
    t, k, b = track(FALL_SEGS, fps=25.0)
    fake = FakeClassifier(onehot7(SITDOWN_))
    assert _falls(drive(_mon(fake), fake, t, k, b)) == []

    # (c) same collapse, classifier says "Fall Down", then >= 0.6 s of horizontal lying: exactly one candidate
    fake = FakeClassifier(onehot7(FALL_))
    falls = _falls(drive(_mon(fake), fake, t, k, b))
    assert len(falls) == 1
    assert 2.5 <= falls[0]["t_start"] <= 3.4 and 3.6 <= falls[0]["t_candidate"] <= 4.6


def test_compute_knee_angle_straight_vs_seated():
    from synth import make_pose
    straight = FM.compute_knee_angle(make_pose(**STAND))
    seated = FM.compute_knee_angle(make_pose(**SIT))
    assert straight > 165.0 and 70.0 < seated < 110.0
    low = make_pose(**STAND); low[[13, 14, 15, 16], 2] = 0.1
    assert np.isnan(FM.compute_knee_angle(low))                   # unreliable legs -> no angle, never a guess


def test_floor_work_and_walking_never_emit_candidates():
    fps = 10.0
    t, k, b = track([hold(LIE, 8.0)], fps=fps)
    k = k.copy(); k[..., 0] += (60.0 * np.sin(2 * np.pi * 2.0 * t))[:, None]       # scrubbing: ~1.6 body-heights/s
    b = b.copy(); b[:, [0, 2]] += (60.0 * np.sin(2 * np.pi * 2.0 * t))[:, None]
    fake = FakeClassifier(onehot7(LYING_))
    assert drive(_mon(fake, eval_every_s=0.5), fake, t, k, b) == []                 # lying-with-motion is floor work
    t, k, b = track([(20.0, dict(STAND, hx=300.0), dict(STAND, hx=900.0))], fps=fps)
    fake = FakeClassifier(onehot7(1))
    assert drive(_mon(fake, eval_every_s=0.5), fake, t, k, b) == []                 # walking person: no immobility either


# ---------------------------------------------------------------- T13
def _lying_feats(angle, aspect, t):
    n = len(t)
    return dict(id=3, t=t, angle=np.asarray(angle, float), aspect=np.asarray(aspect, float), motion=np.zeros(n))


def test_lying_candidate_hold_runs_and_merge_runs_semantics():
    t = np.round(np.arange(0.0, 2.2001, 0.1), 3)
    cond = np.where(t <= 1.0, 1.0, np.where(t < 1.5, np.nan, 1.0))
    runs = FM.hold_runs(t, cond, 1.0)
    assert runs == [(0.0, 2.2)] and FM.merge_runs([(0.0, 1.0), (3.0, 4.0)], 5.0) == [(0.0, 4.0)]

    ang = np.where((t > 1.0) & (t < 1.5), 27.0, 60.0)               # 27 deg / 0.4 -> inside the hysteresis band (NaN)
    asp = np.where((t > 1.0) & (t < 1.5), 0.4, 1.5)
    ev = FM.detect_lying(_lying_feats(ang, asp, t), FM.CFG)
    assert [e["type"] for e in ev] == ["lying"] and ev[0]["t_start"] == 0.0 and np.isclose(ev[0]["t_candidate"], 2.0)

    ang2 = np.where(t > 1.0, 5.0, 60.0); asp2 = np.where(t > 1.0, 0.2, 1.5)   # clearly upright -> run ends at 1.0 s
    assert FM.detect_lying(_lying_feats(ang2, asp2, t), FM.CFG) == []


# ---------------------------------------------------------------- T14
def _hist(fps=25.0, dur=2.4, moving=(), jitter_amp=1.5, conf_joints=None):
    t = np.arange(0, dur, 1 / fps)
    from synth import make_pose
    base = make_pose(**STAND)
    kp = np.repeat(base[None], len(t), axis=0)
    rng = np.random.default_rng(1)
    kp[..., :2] += rng.uniform(-jitter_amp, jitter_amp, size=kp[..., :2].shape)
    for j in moving:
        kp[:, j, 0] += 80.0 * (t - t[0]) / 2.0
    mask = np.ones((len(t), 17), dtype=bool)
    if conf_joints is not None:
        mask[:] = False; mask[:, list(conf_joints)] = True
    return t, kp, mask


def test_windowed_displacement_immobility_ignores_jitter_and_imputed_joints():
    ref = float(np.hypot(50, 180))
    t, kp, mask = _hist()
    d = FM.compute_windowed_displacement(t, kp, mask, ref_scale=ref, win_s=2.0, min_valid_frames=3)
    assert d < 0.02 <= 0.035
    t, kp, mask = _hist(moving=(9, 10, 5), jitter_amp=0.0, conf_joints=(9, 10, 5))   # 14 joints forward-filled
    d = FM.compute_windowed_displacement(t, kp, mask, ref_scale=ref, win_s=2.0, min_valid_frames=3)
    assert d > 0.035
    # mirrored: the only confident joint is still, the moving joint is imputed (mask False) -> still
    t, kp, mask = _hist(moving=(5,), jitter_amp=0.0, conf_joints=(9, 10, 11))
    assert FM.compute_windowed_displacement(t, kp, mask, ref_scale=ref, win_s=2.0, min_valid_frames=3) < 0.01
    # motion older than win_s must be ignored
    t, kp, mask = _hist(dur=4.0, jitter_amp=0.0)
    kp[t < 1.5, :, 0] += (100.0 * (1.5 - t[t < 1.5]))[:, None]
    assert FM.compute_windowed_displacement(t, kp, mask, ref_scale=ref, win_s=2.0, min_valid_frames=3) < 0.01


def _imm(evs):
    return [e for e in evs if e["type"] == "immobility"]


def test_immobility_durations_post_fall_lying_and_seated():
    fps = 10.0
    # lying, motionless (jitter only): immobility at >= 100 s after the lying episode started
    t, k, b = track([hold(LIE, 105.0)], fps=fps); k = jitter(k)
    fake = FakeClassifier(onehot7(LYING_))
    evs = drive(_mon(fake, eval_every_s=0.5), fake, t, k, b)
    imm = _imm(evs)
    assert len(imm) == 1 and 100.0 - 1e-6 <= imm[0]["t_candidate"] - imm[0]["t_start"] < 105.0
    assert any(e["type"] == "lying" for e in evs)

    # post-fall: 40 s after the fall candidate
    t, k, b = track([hold(STAND, 3.0), (0.3, STAND, LIE), hold(LIE, 50.0)], fps=fps); k = jitter(k)
    fake = FakeClassifier()
    pf = lambda ts: onehot7(STAND_) if ts < 3.0 else (onehot7(FALL_) if ts < 5.3 else onehot7(LYING_))
    evs = drive(_mon(fake, eval_every_s=0.5), fake, t, k, b, probs_fn=pf)
    fall, imm = _falls(evs), _imm(evs)
    assert len(fall) == 1 and len(imm) >= 1
    assert fall[0]["t_candidate"] + 40.0 - 1e-6 <= imm[0]["t_candidate"] <= fall[0]["t_candidate"] + 44.0

    # seated: nothing at 30 s, one candidate only after 150 s
    t, k, b = track([hold(SIT, 155.0)], fps=fps); k = jitter(k)
    fake = FakeClassifier(onehot7(SITTING_)); mon = _mon(fake, eval_every_s=0.5)
    first = drive(mon, fake, t[t < 30.0], k[t < 30.0], b[t < 30.0])
    assert first == []
    rest = [mon.update(float(ti), [1], [bi], [ki]) for ti, ki, bi in zip(t[t >= 30.0], k[t >= 30.0], b[t >= 30.0])]
    imm = _imm([e for r in rest for e in r])
    assert len(imm) == 1 and imm[0]["t_candidate"] - imm[0]["t_start"] >= 150.0 - 1e-6


# ---------------------------------------------------------------- T15
EVENT_KEYS = {"type", "track", "t_start", "t_candidate", "t_alert", "priority", "event_id", "status", "evidence"}
EVIDENCE_KEYS = {"clip_window", "tsstg_probs", "ntu60_top5", "kinematics", "snapshot_path"}


def test_candidate_lifecycle_dedup_event_evidence_and_cli_run(tmp_path, monkeypatch):
    t, k, b = track(FALL_SEGS, fps=25.0)
    frames = [np.zeros((H_, W_, 3), dtype=np.uint8)] * len(t)
    fake = FakeClassifier(onehot7(FALL_))
    mon = _mon(fake, evidence_dir=tmp_path / "ev")
    evs = drive(mon, fake, t, k, b, frames=frames)
    falls = _falls(evs)
    assert len(falls) == 1                                       # cooldown/realert dedupe repeat evaluations
    e = falls[0]
    assert EVENT_KEYS <= set(e) and e["status"] == "candidate" and EVIDENCE_KEYS <= set(e["evidence"])
    assert Path(e["evidence"]["snapshot_path"]).exists()
    json.dumps(e)                                                # JSONL-serialisable
    assert e["priority"] == "high" and e["t_alert"] >= e["t_candidate"]
    assert {"drop", "vdown", "hip_rel", "angle", "aspect", "knee_angle", "disp_win", "motion"} <= set(e["evidence"]["kinematics"])
    cw = e["evidence"]["clip_window"]
    assert np.isclose(cw[0], e["t_start"] - 2.0) and np.isclose(cw[1], e["t_alert"] + 2.0)
    assert len(e["evidence"]["ntu60_top5"]) == 5
    done = mon.review_event(e["event_id"], "confirmed", "verified")
    assert done["status"] == "confirmed" and done["operator_note"] == "verified"
    with pytest.raises(ValueError):
        mon.review_event(e["event_id"], "bogus")
    with pytest.raises(KeyError):
        mon.review_event("nope", "dismissed")

    # CLI run(): synthetic source, fake detector + classifier, JSONL + annotated mp4
    import cv2, ultralytics
    src = tmp_path / "in.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W_, H_))
    for _ in range(5):
        vw.write(np.zeros((H_, W_, 3), dtype=np.uint8))
    vw.release()

    class _Boxes:
        def __init__(self, box):
            self.id = torch.tensor([1.0]); self.xyxy = torch.tensor(box[None], dtype=torch.float32)
        def __len__(self): return 1

    class FakeYOLO:
        def __init__(self, *a, **kw): pass
        def track(self, **kw):
            for i in range(len(t)):
                yield types.SimpleNamespace(boxes=_Boxes(b[i]), keypoints=types.SimpleNamespace(data=torch.tensor(k[i][None])),
                                            orig_img=np.zeros((H_, W_, 3), dtype=np.uint8))

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    fake_run = FakeClassifier(onehot7(FALL_))
    monkeypatch.setattr(FM, "TSSTGClassifier", lambda **kw: fake_run)
    out_jsonl, out_mp4 = tmp_path / "events.jsonl", tmp_path / "annot.mp4"
    a = argparse.Namespace(source=str(src), model="yolo26x-pose.pt", tracker="bytetrack.yaml", config=None, stride=1,
                           imgsz=640, conf_det=0.25, device=None, realert=60.0, show=False, save=str(out_mp4),
                           events=str(out_jsonl), evidence_dir=str(tmp_path / "ev_run"))
    FM.run(a)
    recs = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert [r["type"] for r in recs].count("fall") == 1
    assert EVENT_KEYS <= set(recs[0]) and out_mp4.exists() and out_mp4.stat().st_size > 0
