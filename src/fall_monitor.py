#!/usr/bin/env python3
"""fall_monitor.py - Hybrid pose-based FALL / LYING / IMMOBILITY candidate detector.

Features:
- Ultralytics YOLO26-pose + ByteTrack streaming tracking.
- Outlier-clamped 90th-percentile bounding box diagonal ref_scale (non-decaying).
- Spatial-temporal track stitching (stitch_s = 1.5s).
- 25 Hz / 30-frame window resampling + imputation-jump masking.
- Standalone OpenMMLab MMAction2 RecognizerGCN action recognition with temporal smoothing.
- Two-phase pending fall latch with controlled-action / knee-angle veto and strict post-fall stability.
- Windowed confident-joint excursion displacement immobility (post-fall >= 40s, lying >= 100s, seated >= 150s).
- Candidate event lifecycle (review_event) and structured audit evidence.
"""
import argparse
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import yaml

try:
    from .keypoint_adapter import (
        sanitize_keypoints_17,
        resample_window_25hz,
        build_mmaction2_tensors
    )
    from .tsstg_model import (
        TSSTGClassifier,
        CLASS_NAMES,
        CLASS_NAMES_8,
        ActionClass,
        CLASS_TO_IDX_8,
        NTU60_CLASS_NAMES
    )
except (ImportError, ValueError):
    from keypoint_adapter import (
        sanitize_keypoints_17,
        resample_window_25hz,
        build_mmaction2_tensors
    )
    from tsstg_model import (
        TSSTGClassifier,
        CLASS_NAMES,
        CLASS_NAMES_8,
        ActionClass,
        CLASS_TO_IDX_8,
        NTU60_CLASS_NAMES
    )

ROOT = Path(__file__).resolve().parent.parent

def load_cfg(path=None):
    """Loads configuration from config.yaml as the sole source of truth."""
    target = Path(path) if path else (ROOT / "config.yaml" if (ROOT / "config.yaml").exists() else Path("config.yaml"))
    if not target.exists():
        raise FileNotFoundError(f"Master configuration file not found: {target}")
    cfg = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    # Legacy alias support so tests expecting cfg['events'] continue to work
    if 'event_lifecycle' in cfg and 'events' not in cfg:
        cfg['events'] = cfg['event_lifecycle']
    return cfg


CFG = load_cfg()
THRESHOLDS = CFG  # Backward compatibility alias


# ----------------------------------------------------------------------------------------------------------------------
# Feature helper functions
def _tidx(t):
    return pd.to_datetime(t, unit='s')


def _roll(x, t, win_s, fn='mean'):
    r = pd.Series(x, index=_tidx(t)).rolling(f'{max(int(win_s * 1000), 1)}ms', min_periods=1)
    return getattr(r, fn)().to_numpy()


def compute_knee_angle(kp_17: np.ndarray, conf_kp: Optional[float] = None, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Computes interior angle at the knee (in degrees). Returns np.nan if unconfident."""
    if conf_kp is None:
        c = cfg or CFG
        conf_kp = float(c.get('sampling', {}).get('conf_kp', 0.30))
    kp = np.asarray(kp_17, dtype=np.float32)
    angles = []

    # Left leg: hip 11, knee 13, ankle 15
    if kp[11, 2] >= conf_kp and kp[13, 2] >= conf_kp and kp[15, 2] >= conf_kp:
        v1 = kp[11, :2] - kp[13, :2]
        v2 = kp[15, :2] - kp[13, :2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-4 and n2 > 1e-4:
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    # Right leg: hip 12, knee 14, ankle 16
    if kp[12, 2] >= conf_kp and kp[14, 2] >= conf_kp and kp[16, 2] >= conf_kp:
        v1 = kp[12, :2] - kp[14, :2]
        v2 = kp[16, :2] - kp[14, :2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-4 and n2 > 1e-4:
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    if not angles:
        return float(np.nan)
    return float(np.mean(angles))


def update_ref_scale(
    diag_history: list,
    box: np.ndarray,
    max_growth: Optional[float] = None,
    ref_scale_init: Optional[float] = None,
    max_history: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None
) -> float:
    """Updates rolling bounding-box diagonal history in-place and computes clamped scale."""
    c = cfg or CFG
    s_cfg = c.get('sampling', {})
    if max_growth is None:
        max_growth = float(s_cfg.get('ref_scale_max_growth', 1.15))
    if max_history is None:
        max_history = int(s_cfg.get('ref_scale_max_history', 1500))
    """Updates rolling bounding-box diagonal history in-place and computes clamped scale."""
    box = np.asarray(box, dtype=np.float32)
    w = max(float(box[2] - box[0]), 1.0)
    h = max(float(box[3] - box[1]), 1.0)
    d = float(np.hypot(w, h))
    diag_history.append(d)
    while len(diag_history) > max_history:
        diag_history.pop(0)

    p90 = float(np.percentile(diag_history, 90))
    med = float(np.median(diag_history))
    clamped = min(p90, max_growth * med)

    if ref_scale_init is not None:
        clamped = max(clamped, float(ref_scale_init))

    return float(clamped)


def compute_windowed_displacement(
    t_hist: np.ndarray,
    kp17_hist: np.ndarray,
    conf_mask_hist: np.ndarray,
    ref_scale: float,
    win_s: Optional[float] = None,
    min_valid_frames: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None
) -> float:
    """Computes median windowed centroid excursion across confident joints normalized by scale."""
    c = cfg or CFG
    imm_cfg = c.get('immobility', {})
    if win_s is None:
        win_s = float(imm_cfg.get('win_s', 10.0))
    if min_valid_frames is None:
        min_valid_frames = int(imm_cfg.get('min_valid_frames', 3))
    """Computes median windowed centroid excursion across confident joints normalized by scale."""
    t = np.asarray(t_hist, dtype=np.float64)
    if len(t) == 0:
        return float(np.nan)
    t_now = t[-1]
    sel = (t >= t_now - win_s)
    if not np.any(sel):
        return float(np.nan)

    kps = np.asarray(kp17_hist, dtype=np.float32)[sel]
    masks = np.asarray(conf_mask_hist, dtype=bool)[sel]

    excursions = []
    for j in range(17):
        valid_j = masks[:, j]
        if np.sum(valid_j) >= min_valid_frames:
            pts_j = kps[valid_j, j, :2]
            centroid = np.mean(pts_j, axis=0)
            dists = np.linalg.norm(pts_j - centroid, axis=1)
            excursions.append(float(np.max(dists)))

    if not excursions:
        return float(np.nan)

    med_exc = float(np.median(excursions))
    scale = max(float(ref_scale), 1.0)
    return med_exc / scale


def compute_features(tr, cfg=None, ref_scale=None):
    """Per-track kinematic features preserving baseline formulas."""
    c = cfg or CFG
    s = c['sampling']
    t = tr['t']
    kp = tr['kp']
    box = tr['box']

    xy = kp[..., :2].copy()
    vis = kp[..., 2] >= s['conf_kp']
    xy[~vis] = np.nan

    sh = np.nanmean(xy[:, [5, 6]], axis=1)
    hp = np.nanmean(xy[:, [11, 12]], axis=1)
    head = np.nanmean(xy[:, [0, 1, 2]], axis=1)

    d = sh - hp
    angle = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    w = box[:, 2] - box[:, 0]
    h = np.maximum(box[:, 3] - box[:, 1], 1.0)

    if ref_scale is None:
        rolling_win = f"{int(s.get('ref_scale_rolling_s', 10.0) * 1000)}ms"
        ref_h = pd.Series(h, index=_tidx(t)).rolling(rolling_win, min_periods=1).quantile(0.9).to_numpy()
    else:
        ref_h = np.full(len(t), float(ref_scale))

    hip_rel = (box[:, 3] - hp[:, 1]) / ref_h
    sm = s['smooth_s']

    f = dict(
        id=tr['id'],
        t=t,
        angle=_roll(angle, t, sm),
        aspect=_roll(w / h, t, sm),
        hip_rel=_roll(hip_rel, t, sm),
        head_rel=_roll((box[:, 3] - head[:, 1]) / ref_h, t, sm),
        visibility=vis.mean(axis=1)
    )

    dt_min = float(s.get('dt_min', 0.001))
    dt = np.maximum(np.diff(t), dt_min)
    v = np.full(len(t), np.nan)
    v[1:] = -np.diff(f['hip_rel']) / dt  # downward = positive

    mo = np.full(len(t), np.nan)
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=2)
    mo[1:] = np.nanmean(step, axis=1) / ref_h[1:] / dt
    mo[1:][np.diff(t) > c['lying']['hold_s']] = np.nan

    f['vdown'] = v
    f['motion'] = _roll(mo, t, float(s.get('motion_smooth_s', 1.0)))
    return f


def hold_runs(t, cond, hold_s):
    """Condition hold run extractor with gap tolerance."""
    runs, start, last_true, prev = [], None, None, None
    for ti, c in zip(t, cond):
        if start is not None and prev is not None and ti - prev > hold_s:
            runs.append((start, last_true))
            start = None
        if c == 1:
            if start is None:
                start = ti
            last_true = ti
        elif c == 0:
            if start is not None:
                runs.append((start, last_true))
                start = None
        elif start is not None and ti - last_true > hold_s:
            runs.append((start, last_true))
            start = None
        prev = ti
    if start is not None:
        runs.append((start, last_true))
    return runs


def merge_runs(runs, gap):
    """Merges consecutive intervals separated by <= gap."""
    out = []
    for s, e in runs:
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _mean_in(x, t, s, e):
    m = (t >= s) & (t <= e)
    v = x[m]
    return float(np.nanmean(v)) if m.any() and np.isfinite(v).any() else 0.0


def lying_cond(f, p):
    ang, asp = f['angle'], f['aspect']
    mult = float(p.get('aspect_in_nan_mult', 1.3))
    ein = np.where(np.isnan(ang), asp > p['aspect_in'] * mult, (ang > p['angle_in']) & (asp > p['aspect_in']))
    eout = np.where(np.isnan(ang), asp < p['aspect_out'], (ang < p['angle_out']) | (asp < p['aspect_out']))
    return np.where(ein, 1.0, np.where(eout, 0.0, np.nan))


def detect_lying(f, cfg=None):
    c = cfg or CFG
    p = c['lying']
    t = f['t']
    cooldown = float(c.get('event_lifecycle', {}).get('cooldown_s', c.get('events', {}).get('cooldown_s', 5.0)))
    runs = merge_runs(hold_runs(t, lying_cond(f, p), p['hold_s']), cooldown)
    ev = []
    for s, e in runs:
        if e - s >= p['T_lying'] and _mean_in(f['motion'], t, s, e) <= p['max_motion']:
            ev.append(dict(
                type='lying',
                track=f['id'],
                t_start=float(s),
                t_candidate=float(s + p['T_lying']),
                priority='normal'
            ))
    return ev


def detect_falls(f, cfg=None):
    """Baseline kinematic fall detector for cold-start / rule evaluation."""
    c = cfg or CFG
    p = c['fall']
    t = f['t']
    hr = f['hip_rel']
    min_obs = int(c.get('sampling', {}).get('min_track_observations', 3))
    if len(t) < min_obs:
        return []

    win = f"{int(p['window_s'] * 1000)}ms"
    idx = _tidx(t)
    drop = (pd.Series(hr, index=idx).rolling(win, min_periods=2).max() - pd.Series(hr, index=idx)).to_numpy()
    vpk = pd.Series(f['vdown'], index=idx).rolling(win, min_periods=1).max().to_numpy()
    hit = (drop > p['drop']) & (vpk > p['vdown'])

    cooldown = float(c.get('event_lifecycle', {}).get('cooldown_s', c.get('events', {}).get('cooldown_s', 5.0)))
    ev, last = [], -1e9
    for i in np.flatnonzero(hit):
        if t[i] - last < cooldown:
            continue
        sel = (t >= t[i]) & (t <= t[i] + p['post_s'])
        ts = t[sel]
        low = hr[sel] < p['low_hip']
        horiz = (f['angle'][sel] > p['angle']) | (f['aspect'][sel] > p['aspect']) | (f['head_rel'][sel] < p.get('head_low', 0.0))
        cond = np.where(low & horiz, 1.0, np.where(np.isnan(hr[sel]), np.nan, 0.0))

        for s, e in hold_runs(ts, cond, p['hold_s']):
            if e - s >= p['stable_s'] and _mean_in(f['motion'], t, s, e) <= p['max_motion']:
                w = (t >= t[i] - p['window_s']) & (t <= t[i])
                onset = t[w][np.nanargmax(hr[w])] if np.isfinite(hr[w]).any() else t[i]
                ev.append(dict(
                    type='fall',
                    track=f['id'],
                    t_start=float(onset),
                    t_candidate=float(s + p['stable_s']),
                    priority='high'
                ))
                last = t[i]
                break
    return ev


# ----------------------------------------------------------------------------------------------------------------------
# BaselineFallMonitor preserving exact S0 implementation
class BaselineFallMonitor:
    def __init__(self, cfg=CFG, buffer_s=30.0, eval_every_s=0.5, realert_s=60.0):
        self.cfg = cfg
        self.buffer_s = buffer_s
        self.every = eval_every_s
        self.realert_s = realert_s
        self.tracks, self.last_eval, self.emitted, self.last_cand = {}, {}, {}, {}

    def update(self, t, ids, boxes, kps):
        for tid, b, k in zip(ids, boxes, kps):
            d = self.tracks.setdefault(int(tid), dict(t=[], kp=[], box=[]))
            d['t'].append(float(t))
            d['kp'].append(k)
            d['box'].append(b)
            while d['t'][0] < t - self.buffer_s:
                for key in d:
                    d[key].pop(0)

        for tid in [i for i, d in self.tracks.items() if t - d['t'][-1] > self.buffer_s]:
            del self.tracks[tid]
            self.last_eval.pop(tid, None)

        new = []
        for tid in dict.fromkeys(int(i) for i in ids):
            d = self.tracks[tid]
            if len(d['t']) < 5 or t - self.last_eval.get(tid, -1e9) < self.every:
                continue
            self.last_eval[tid] = t
            f = compute_features(dict(id=tid, t=np.array(d['t']), kp=np.array(d['kp']), box=np.array(d['box'])), self.cfg)
            for ev in detect_falls(f, self.cfg) + detect_lying(f, self.cfg):
                key = (ev['type'], tid)
                if ev['t_candidate'] > t + 1e-6 or t - self.emitted.get(key, -1e9) < self.realert_s:
                    continue
                if ev['t_candidate'] <= self.last_cand.get(key, -1e9) + self.cfg['events']['cooldown_s']:
                    continue
                self.emitted[key] = t
                self.last_cand[key] = ev['t_candidate']
                new.append(dict(ev, t_alert=float(t)))
        return new


# ----------------------------------------------------------------------------------------------------------------------
# Upgraded FallMonitor streaming engine
class FallMonitor:
    def __init__(
        self,
        cfg=None,
        buffer_s: Optional[float] = None,
        eval_every_s: Optional[float] = None,
        realert_s: Optional[float] = None,
        stitch_s: Optional[float] = None,
        classifier=None,
        evidence_dir: Optional[Union[str, Path]] = None,
        img_shape: Optional[Tuple[int, int]] = None,
        weight_dir: Union[str, Path] = "weights",
        variant: str = "mmaction2",
        streams: Optional[Sequence[str]] = None,
        device: str = "cpu",
        trace: bool = False
    ):
        self.trace: Optional[List[Dict[str, Any]]] = [] if trace else None
        base_cfg = load_cfg()
        if cfg is not None:
            merged_cfg = copy.deepcopy(base_cfg)
            for k, v in cfg.items():
                if isinstance(v, dict) and k in merged_cfg and isinstance(merged_cfg[k], dict):
                    merged_cfg[k].update(v)
                else:
                    merged_cfg[k] = v
            self.cfg = merged_cfg
        else:
            self.cfg = copy.deepcopy(base_cfg)
        if 'event_lifecycle' in self.cfg and 'events' not in self.cfg:
            self.cfg['events'] = self.cfg['event_lifecycle']

        s_cfg = self.cfg.get('sampling', {})
        el_cfg = self.cfg.get('event_lifecycle', {})
        v_cfg = self.cfg.get('video', {})
        d_cfg = self.cfg.get('detector', {})

        self.buffer_s = buffer_s if buffer_s is not None else float(s_cfg.get('buffer_s', 30.0))
        self.every = eval_every_s if eval_every_s is not None else float(s_cfg.get('eval_every_s', 0.50))
        self.realert_s = realert_s if realert_s is not None else float(el_cfg.get('realert_s', 60.0))
        self.stitch_s = stitch_s if stitch_s is not None else float(el_cfg.get('stitch_s', 3.0))

        default_h = int(v_cfg.get('default_height', 240))
        default_w = int(v_cfg.get('default_width', 320))
        self.img_shape = img_shape if img_shape is not None else (default_h, default_w)
        self.device = device

        chosen_streams = streams if streams is not None else tuple(d_cfg.get('streams', ('j', 'jm')))

        if classifier is not None:
            self.classifier = classifier
        else:
            try:
                proj_cfg = self.cfg.get('action_classes', {}).get('projection')
                self.classifier = TSSTGClassifier(
                    weight_dir=weight_dir, variant=variant, streams=chosen_streams, num_classes=8, device=device,
                    projection_cfg=proj_cfg
                )
            except Exception:
                self.classifier = None

        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.tracks: Dict[int, Dict[str, Any]] = {}
        action_names = self.cfg.get('action_classes', {}).get('names', [
            "Standing", "Walking", "Sitting", "Lying", "StandUp", "SitDown", "Fall", "Instability"
        ])
        self.action_class_map = {name.lower(): i for i, name in enumerate(action_names)}
        self.lost_tracks: Dict[int, Dict[str, Any]] = {}
        self.last_eval: Dict[int, float] = {}
        self.emitted: Dict[Tuple[str, int], float] = {}
        self.last_cand: Dict[Tuple[str, int], float] = {}
        self.events: Dict[str, Dict[str, Any]] = {}


    def _trace_row(self, t, tid, p, has_win, pred_idx, probs_7, probs_s, drop, vpk, hr, ang, asp, head, motion,
                   knee, kin_veto, veto, pa, pb, pc, d, f):
        horiz = (ang > p['angle']) or (asp > p['aspect']) or (head < p.get('head_low', 0.0))
        fall_i = self.action_class_map.get('fall', 6)
        stand_i = self.action_class_map.get('standing', 0)
        walk_i = self.action_class_map.get('walking', 1)
        sit_i = self.action_class_map.get('sitting', 2)
        sitd_i = self.action_class_map.get('sitdown', 5)
        self.trace.append(dict(
            t=float(t), track=int(tid), has_win=bool(has_win), pred=int(pred_idx),
            p_fall=float(probs_7[fall_i]) if len(probs_7) > fall_i else 0.0,
            p_fall_s=float(probs_s[fall_i]) if len(probs_s) > fall_i else 0.0,
            p_stand=float(probs_s[stand_i]) if len(probs_s) > stand_i else 0.0,
            p_walk=float(probs_s[walk_i]) if len(probs_s) > walk_i else 0.0,
            p_sit=float(probs_s[sit_i]) if len(probs_s) > sit_i else 0.0,
            p_sitdown=float(probs_s[sitd_i]) if len(probs_s) > sitd_i else 0.0,
            drop=float(drop), vpk=float(vpk), hip_rel=float(hr), angle=float(ang), aspect=float(asp),
            head_rel=float(head), motion=float(motion), knee=float(knee), vis=float(f['visibility'][-1]),
            kin_dyn=bool(drop > p['drop'] and vpk > p['vdown']),
            kin_post=bool(hr < p['low_hip'] and horiz),
            kin_veto=bool(kin_veto), veto=bool(veto),
            path_a=bool(pa), path_b=bool(pb), path_c=bool(pc),
            pending=d['pending_fall_t0'] is not None))

    def review_event(self, event_id: str, status: str, operator_note: str = "") -> Dict[str, Any]:
        """Operator review API transitioning candidate status."""
        if event_id not in self.events:
            raise KeyError(f"Event ID not found: {event_id}")
        allowed = {"acknowledged", "confirmed", "dismissed"}
        if status not in allowed:
            raise ValueError(f"Invalid status '{status}'. Must be one of {allowed}")

        record = self.events[event_id]
        record["status"] = status
        record["operator_note"] = operator_note
        return record

    def update(
        self,
        t: float,
        ids: Sequence[int],
        boxes: Sequence[np.ndarray],
        kps: Sequence[np.ndarray],
        frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        t = float(t)

        # 1. Update lost tracks: any track in self.tracks not seen in current ids
        active_ids = {int(i) for i in ids}
        for tid in list(self.tracks.keys()):
            if tid not in active_ids:
                lost_tr = self.tracks.pop(tid)
                self.lost_tracks[tid] = dict(
                    last_t=lost_tr['t'][-1] if lost_tr['t'] else t,
                    last_box=lost_tr['box'][-1],
                    ref_scale=lost_tr['ref_scale'],
                    data=lost_tr
                )

        # 2. Spatial-temporal track stitching
        for tid, b in zip(ids, boxes):
            tid = int(tid)
            if tid in self.lost_tracks:
                # Same ID restored from brief occlusion
                inherited = self.lost_tracks.pop(tid)
                self.tracks[tid] = inherited["data"]
            elif tid not in self.tracks and self.stitch_s > 0:
                box_center = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
                best_match = None
                best_dist = 1e9

                for lost_id, lost in list(self.lost_tracks.items()):
                    gap = t - lost["last_t"]
                    if 0 <= gap <= self.stitch_s:
                        lost_box = lost["last_box"]
                        lost_center = np.array([(lost_box[0] + lost_box[2]) / 2.0, (lost_box[1] + lost_box[3]) / 2.0])
                        dist = np.linalg.norm(box_center - lost_center)
                        max_stitch_dist = float(self.cfg.get('event_lifecycle', {}).get('stitch_max_dist_ref_scale', 0.8)) * lost["ref_scale"]
                        if dist <= max_stitch_dist and dist < best_dist:
                            best_match = lost_id
                            best_dist = dist

                if best_match is not None:
                    # Inherit state and history
                    inherited = self.lost_tracks.pop(best_match)
                    self.tracks[tid] = inherited["data"]
                    self.tracks[tid]["id"] = tid
                    if best_match in self.last_eval:
                        self.last_eval[tid] = self.last_eval.pop(best_match)

        # 3. Append observations
        for tid, b, k in zip(ids, boxes, kps):
            tid = int(tid)
            d = self.tracks.setdefault(tid, dict(
                id=tid,
                t=[], kp=[], conf_mask=[], box=[], diag_hist=[],
                ref_scale=float(np.hypot(b[2] - b[0], b[3] - b[1])),
                pending_fall_t0=None,
                post_fall_stable_s=0.0,
                last_fall_cand_t=None,
                lying_start_t=None,
                seated_start_t=None,
                probs_hist=[]
            ))

            prev_kp = d['kp'][-1] if d['kp'] else None
            clean_kp, is_conf = sanitize_keypoints_17(k, b, prev_kp, conf_kp=self.cfg['sampling']['conf_kp'])

            d['t'].append(t)
            d['kp'].append(clean_kp)
            d['conf_mask'].append(is_conf)
            d['box'].append(b)

            ref = update_ref_scale(d['diag_hist'], b, ref_scale_init=d.get('ref_scale'))
            d['ref_scale'] = ref

            # Trim sliding buffer
            while len(d['t']) > 1 and d['t'][0] < t - self.buffer_s:
                d['t'].pop(0)
                d['kp'].pop(0)
                d['conf_mask'].pop(0)
                d['box'].pop(0)

        # Purge old lost_tracks
        purge_s = float(self.cfg.get('event_lifecycle', {}).get('purge_lost_tracks_s', 10.0))
        for lost_id in [i for i, tr in self.lost_tracks.items() if t - tr['last_t'] > max(self.stitch_s, purge_s)]:
            self.lost_tracks.pop(lost_id, None)

        # 4. Evaluate candidates per track
        new_events = []
        min_obs = int(self.cfg.get('sampling', {}).get('min_track_observations', 3))
        for tid in dict.fromkeys(int(i) for i in ids):
            d = self.tracks[tid]
            if len(d['t']) < min_obs or (t - self.last_eval.get(tid, -1e9) < self.every):
                continue
            self.last_eval[tid] = t

            f = compute_features(
                dict(id=tid, t=np.array(d['t']), kp=np.array(d['kp']), box=np.array(d['box'])),
                self.cfg,
                ref_scale=d['ref_scale']
            )

            p_fall = self.cfg['fall']
            hr = f['hip_rel']
            idx = _tidx(d['t'])
            win = f"{int(p_fall['window_s'] * 1000)}ms"
            drop_s = (pd.Series(hr, index=idx).rolling(win, min_periods=2).max() - pd.Series(hr, index=idx)).to_numpy()
            vpk_s = pd.Series(f['vdown'], index=idx).rolling(win, min_periods=1).max().to_numpy()

            curr_drop = float(drop_s[-1]) if len(drop_s) > 0 and np.isfinite(drop_s[-1]) else 0.0
            curr_vpk = float(vpk_s[-1]) if len(vpk_s) > 0 and np.isfinite(vpk_s[-1]) else 0.0
            curr_hr = float(hr[-1]) if np.isfinite(hr[-1]) else 1.0
            curr_angle = float(f['angle'][-1]) if np.isfinite(f['angle'][-1]) else 0.0
            curr_aspect = float(f['aspect'][-1]) if np.isfinite(f['aspect'][-1]) else 0.0
            curr_head = float(f['head_rel'][-1]) if np.isfinite(f['head_rel'][-1]) else 0.5
            curr_motion = float(f['motion'][-1]) if np.isfinite(f['motion'][-1]) else 0.0
            knee_angle = compute_knee_angle(d['kp'][-1], conf_kp=self.cfg['sampling']['conf_kp'])

            # Window resampling and action prediction
            w_cfg = self.cfg.get('windowing', {})
            norm_cfg = self.cfg.get('normalization', {})
            resampled = resample_window_25hz(
                d['t'], d['kp'], d['conf_mask'],
                window_s=float(p_fall.get('window_s', w_cfg.get('primary_window_s', 2.5))),
                num_frames=int(p_fall.get('num_frames', w_cfg.get('clip_len', 100))),
                min_window_s=float(w_cfg.get('min_window_s', 1.16)),
                max_gap_s=float(w_cfg.get('max_gap_s', p_fall.get('hold_s', 1.0))),
                min_observations=int(w_cfg.get('min_observations_for_resample', 6))
            )

            has_win = (resampled is not None and self.classifier is not None)
            pred_idx = 0
            num_classes = len(self.cfg.get('action_classes', {}).get('names', [])) or 8
            probs_7 = np.zeros(num_classes, dtype=np.float64)
            ntu_probs = np.zeros(60, dtype=np.float64)

            if has_win:
                res_pts, res_mask = resampled
                stream_names = getattr(self.classifier, 'streams', ('j', 'jm'))
                stream_tensors = build_mmaction2_tensors(
                    res_pts,
                    conf_mask_30_17=res_mask,
                    img_shape=self.img_shape,
                    streams=stream_names,
                    norm_mode=norm_cfg.get('mode', 'prenormalize2d'),
                    clip_len=int(w_cfg.get('clip_len', 100)),
                    compact_scale_factor=float(norm_cfg.get('track_compact_scale_factor', 1.25))
                )
                pred_idx, _, probs_7, ntu_probs = self.classifier(stream_tensors)

            d['probs_hist'].append((t, probs_7))
            retention_s = float(self.cfg.get('sampling', {}).get('probs_hist_retention_s', 1.0))
            while d['probs_hist'] and d['probs_hist'][0][0] < t - retention_s:
                d['probs_hist'].pop(0)

            # Temporal probability smoothing over smooth_s (padded to homogeneous shape)
            smooth_s = float(self.cfg.get('sampling', {}).get('smooth_s', 0.40))
            smooth_win = [
                (np.pad(p, (0, num_classes - len(p))) if len(p) < num_classes else p[:num_classes])
                for pt, p in d['probs_hist'] if pt >= t - smooth_s
            ]
            probs_smooth = np.mean(smooth_win, axis=0) if smooth_win else (np.pad(probs_7, (0, num_classes - len(probs_7))) if len(probs_7) < num_classes else probs_7[:num_classes])

            # Kinematic veto — computed always from config
            kv_bend = self.cfg['kinematic_veto']['controlled_bend']
            controlled_bend = (
                curr_angle < kv_bend['max_angle'] and
                curr_aspect < kv_bend['max_aspect'] and
                curr_head > kv_bend['min_head_rel'] and
                curr_vpk <= kv_bend['max_vdown']
            )
            kv_knee = self.cfg['kinematic_veto']['controlled_knee']
            controlled_knee = (
                not np.isnan(knee_angle) and
                (kv_knee['min_knee_angle'] <= knee_angle <= kv_knee['max_knee_angle']) and
                curr_angle < kv_knee['max_angle'] and
                curr_aspect < kv_knee['max_aspect'] and
                curr_hr >= kv_knee['min_hip_rel']
            )
            kinematic_veto = bool(controlled_bend or controlled_knee)

            # Action veto — only available when classifier has a full window and posture is upright
            controlled_veto = kinematic_veto  # kinematic veto active even in cold-start
            fall_idx = self.action_class_map.get('fall', 6)
            unstable_idx = self.action_class_map.get('instability', 7)
            lying_idx = self.action_class_map.get('lying', 3)
            p_fall_act = float(probs_smooth[fall_idx]) if len(probs_smooth) > fall_idx else 0.0
            p_unstable_act = float(probs_smooth[unstable_idx]) if len(probs_smooth) > unstable_idx else 0.0
            p_lying_act = float(probs_smooth[lying_idx]) if len(probs_smooth) > lying_idx else 0.0

            if has_win:
                kv_up = self.cfg['kinematic_veto']['upright_posture']
                is_upright = (
                    curr_angle < kv_up['max_angle'] and
                    curr_aspect < kv_up['max_aspect'] and
                    curr_head > kv_up['min_head_rel']
                )
                upright_indices = self.cfg.get('action_classes', {}).get('projection', {}).get('upright_action_indices', [0, 1, 2, 4, 5])
                controlled_action = bool(
                    is_upright and
                    (np.max(probs_smooth[upright_indices]) > np.max([p_lying_act, p_fall_act]))
                )
                controlled_veto = bool(controlled_action or kinematic_veto)

            raw_cands = []

            # ------------------------------------------------------------------
            # Fall Latch Evaluation
            p_a = self.cfg['path_a']
            p_b = self.cfg['path_b']
            p_c = self.cfg['path_c']

            if d['pending_fall_t0'] is None:
                # Path A: ST-GCN Fall Trigger (A43 only, A42 cannot trigger alone)
                fall_cls_idx = self.action_class_map.get('fall', 6)
                path_a = (
                    has_win and pred_idx == fall_cls_idx and
                    p_fall_act >= p_a['min_action_score'] and
                    (not controlled_veto) and
                    (curr_drop > p_a['min_drop'] or curr_vpk > p_a['min_vdown']) and
                    (curr_angle > p_a['trigger_angle'] or curr_aspect > p_a['trigger_aspect'] or curr_hr < p_a['trigger_low_hip'])
                )
                # Path B: Kinematic Collapse with Action Support (Fall or Lying)
                action_supports_fall = (
                    (p_fall_act >= p_b['support_fall_score']) or
                    (p_lying_act + p_fall_act >= p_b.get('combined_support_score', 0.25)) or
                    (p_unstable_act >= p_b.get('support_instability_score', 0.35))
                )
                path_b = (
                    has_win and curr_drop > p_b['drop'] and curr_vpk > p_b['vdown'] and
                    (not controlled_veto) and
                    action_supports_fall and
                    curr_hr < p_b['low_hip'] and
                    (curr_angle > p_b['angle'] or curr_aspect > p_b['aspect'] or curr_head < p_b['head_low'])
                )
                # Path C: Cold-Start Fallback (< min_window_s or missing classifier)
                path_c = (
                    (not has_win) and
                    curr_drop > p_c['drop'] and curr_vpk > p_c['vdown'] and
                    curr_hr < p_c['low_hip'] and
                    (curr_angle > p_c['angle'] or curr_aspect > p_c['aspect'] or curr_head < p_c['head_low'])
                )

                if path_a or path_b or path_c:
                    d['pending_fall_t0'] = t
                if self.trace is not None:
                    self._trace_row(t, tid, p_fall, has_win, pred_idx, probs_7, probs_smooth, curr_drop, curr_vpk,
                                    curr_hr, curr_angle, curr_aspect, curr_head, curr_motion, knee_angle,
                                    kinematic_veto, controlled_veto, path_a, path_b, path_c, d, f)

            p_conf = self.cfg['post_fall_confirmation']
            if d['pending_fall_t0'] is not None:
                if t - d['pending_fall_t0'] <= p_conf['post_s']:
                    arr_t = f['t']
                    win_s_fall = float(self.cfg['fall']['window_s'])
                    w_sel = (arr_t >= d['pending_fall_t0'] - win_s_fall) & (arr_t <= d['pending_fall_t0'])
                    onset_t = float(arr_t[w_sel][np.nanargmax(hr[w_sel])]) if np.any(w_sel) and np.isfinite(hr[w_sel]).any() else float(d['pending_fall_t0'])

                    sel_post = (arr_t >= onset_t) & (arr_t <= d['pending_fall_t0'] + p_conf['post_s'])
                    if np.any(sel_post):
                        ts_post = arr_t[sel_post]
                        hr_post = f['hip_rel'][sel_post]
                        ang_post = f['angle'][sel_post]
                        asp_post = f['aspect'][sel_post]
                        head_post = f['head_rel'][sel_post]

                        # Upright chair sitting rejection
                        chair_rej = p_conf['chair_rejection']
                        is_upright_sitting = (ang_post < chair_rej['max_angle']) & (asp_post < chair_rej['max_aspect'])

                        # Mode 1: Standard horizontal recumbence
                        std_posture = (
                            (hr_post <= p_conf['std_low_hip']) &
                            (
                                (ang_post >= p_conf['std_angle']) |
                                (asp_post >= p_conf['std_aspect'])
                            )
                        )
                        # Mode 2: Floor-curled / floor-seated collapse
                        floor_posture = (
                            (hr_post <= p_conf['floor_low_hip']) &
                            (curr_drop >= p_conf['floor_min_drop']) &
                            (
                                (ang_post >= p_conf['floor_angle']) |
                                (asp_post >= p_conf['floor_aspect']) |
                                (head_post <= p_conf['floor_head_low'])
                            )
                        )
                        # Mode 3: Low-seated or sofa/bed slump
                        sofa_posture = (
                            (hr_post <= p_conf['sofa_low_hip']) &
                            (curr_drop >= p_conf['sofa_min_drop']) &
                            (
                                (ang_post >= p_conf['sofa_angle']) |
                                (asp_post >= p_conf['sofa_aspect'])
                            )
                        )

                        is_confirmed_posture = (std_posture | floor_posture | sofa_posture) & (~is_upright_sitting)
                        cond_post = np.where(is_confirmed_posture & (not kinematic_veto), 1.0, np.where(np.isnan(hr_post), np.nan, 0.0))
                        runs_post = hold_runs(ts_post, cond_post, p_conf['hold_s'])
                        max_mot_allowed = p_conf['max_motion']

                        for s_run, e_run in runs_post:
                            if (e_run - s_run >= p_conf['stable_s']) and (_mean_in(f['motion'], arr_t, s_run, e_run) <= max_mot_allowed):
                                t_cand = float(s_run + p_conf['stable_s'])
                                raw_cands.append(dict(
                                    type='fall',
                                    track=tid,
                                    t_start=onset_t,
                                    t_candidate=t_cand,
                                    priority='high'
                                ))
                                d['last_fall_cand_t'] = t_cand
                                d['pending_fall_t0'] = None
                                break
                else:
                    # Latch expired without sustained floor recumbence:
                    # If person recovers upright within post_s, emit 'fall_recovered'
                    ev_lc = self.cfg['event_lifecycle']
                    if curr_angle < ev_lc['recovery_max_angle'] and curr_hr > ev_lc['recovery_min_hip']:
                        if ev_lc.get('allow_immediate_recovery', True):
                            raw_cands.append(dict(
                                type='fall_recovered',
                                track=tid,
                                t_start=float(d['pending_fall_t0']),
                                t_candidate=float(t),
                                priority='low',
                                status='recovered'
                            ))
                    d['pending_fall_t0'] = None

            # ------------------------------------------------------------------
            # Lying Detection
            for lev in detect_lying(f, self.cfg):
                raw_cands.append(lev)

            # ------------------------------------------------------------------
            # Immobility Detection (Windowed Excursion)
            disp_win = compute_windowed_displacement(
                np.array(d['t']), np.array(d['kp']), np.array(d['conf_mask']),
                ref_scale=d['ref_scale'],
                win_s=self.cfg['immobility']['win_s'],
                min_valid_frames=self.cfg['immobility']['min_valid_frames']
            )

            is_still = np.isfinite(disp_win) and (disp_win <= self.cfg['immobility']['disp_thresh'])
            imm_cfg = self.cfg['immobility']

            if is_still:
                # 1. Post-fall immobility
                if d['last_fall_cand_t'] is not None:
                    rec_cfg = imm_cfg.get('post_fall_recovery', {})
                    min_hip_rec = float(rec_cfg.get('min_hip', 0.40))
                    max_angle_rec = float(rec_cfg.get('max_angle', 35.0))
                    if curr_hr > min_hip_rec or curr_angle < max_angle_rec:
                        # Person has recovered to an upright stance
                        d['last_fall_cand_t'] = None
                    elif t >= d['last_fall_cand_t'] + float(imm_cfg.get('T_immobile_post_fall', 40.0)):
                        raw_cands.append(dict(
                            type='immobility',
                            track=tid,
                            t_start=float(d['last_fall_cand_t']),
                            t_candidate=float(t),
                            priority='normal'
                        ))

                # Posture classification for stillness timers
                lp_cfg = imm_cfg['lying_posture']
                is_lying_posture = (curr_angle >= lp_cfg['min_angle']) or (
                    curr_aspect >= lp_cfg['min_aspect'] and curr_angle >= lp_cfg['min_angle_relaxed']
                )

                sp_cfg = imm_cfg['seated_posture']
                sit_cls_idx = self.action_class_map.get('sitting', 2)
                is_seated_posture = (not is_lying_posture) and (
                    (not np.isnan(knee_angle) and sp_cfg['min_knee_angle'] <= knee_angle <= sp_cfg['max_knee_angle']) or
                    (pred_idx == sit_cls_idx or probs_smooth[sit_cls_idx] > sp_cfg['min_prob_sit']) or
                    (curr_angle <= sp_cfg['max_angle'] and curr_aspect <= sp_cfg['max_aspect'])
                )

                # 2. Lying stillness immobility
                if is_lying_posture:
                    if d['lying_start_t'] is None:
                        d['lying_start_t'] = t
                    elif t - d['lying_start_t'] >= imm_cfg['T_immobile_lying']:
                        raw_cands.append(dict(
                            type='immobility',
                            track=tid,
                            t_start=float(d['lying_start_t']),
                            t_candidate=float(t),
                            priority='normal'
                        ))
                else:
                    d['lying_start_t'] = None

                # 3. Seated stillness immobility
                if is_seated_posture:
                    if d['seated_start_t'] is None:
                        d['seated_start_t'] = t
                    elif t - d['seated_start_t'] >= imm_cfg['T_immobile_seated']:
                        raw_cands.append(dict(
                            type='immobility',
                            track=tid,
                            t_start=float(d['seated_start_t']),
                            t_candidate=float(t),
                            priority='normal'
                        ))
                else:
                    d['seated_start_t'] = None
            else:
                # Active movement resets stillness timers and clears post-fall immobility on recovery
                d['lying_start_t'] = None
                d['seated_start_t'] = None
                rec_cfg = imm_cfg.get('post_fall_recovery', {})
                motion_clear = float(rec_cfg.get('active_motion_clear', 1.20))
                hip_clear = float(rec_cfg.get('active_hip_clear', 0.35))
                if curr_motion > motion_clear or curr_hr > hip_clear:
                    d['last_fall_cand_t'] = None

            # ------------------------------------------------------------------
            # Deduplication & Evidence Packaging
            for ev in raw_cands:
                key = (ev['type'], tid)
                if ev['t_candidate'] > t + 1e-6 or t - self.emitted.get(key, -1e9) < self.realert_s:
                    continue
                if ev['t_candidate'] <= self.last_cand.get(key, -1e9) + self.cfg['events']['cooldown_s']:
                    continue

                self.emitted[key] = t
                self.last_cand[key] = ev['t_candidate']
                event_id = uuid.uuid4().hex[:12]

                # Save snapshot if evidence directory configured
                snap_path = ""
                if self.evidence_dir:
                    snap_file = self.evidence_dir / f"{event_id}.jpg"
                    if frame is not None and frame.size > 0:
                        cv2.imwrite(str(snap_file), frame)
                    else:
                        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                        cv2.imwrite(str(snap_file), dummy)
                    snap_path = str(snap_file)

                # NTU60 top 5 classes
                top5_indices = np.argsort(ntu_probs)[-5:][::-1]
                ntu60_top5 = [
                    (NTU60_CLASS_NAMES[i] if i < len(NTU60_CLASS_NAMES) else str(i), float(ntu_probs[i]))
                    for i in top5_indices
                ]

                # Kinematics summary
                kinematics = dict(
                    drop=float(curr_drop),
                    vdown=float(curr_vpk),
                    hip_rel=float(curr_hr),
                    angle=float(curr_angle),
                    aspect=float(curr_aspect),
                    knee_angle=float(knee_angle) if not np.isnan(knee_angle) else 0.0,
                    disp_win=float(disp_win),
                    motion=float(curr_motion)
                )

                evidence = dict(
                    clip_window=[float(ev['t_start'] - 2.0), float(t + 2.0)],
                    tsstg_probs={name: float(probs_7[i]) for i, name in enumerate(CLASS_NAMES)},
                    ntu60_top5=ntu60_top5,
                    kinematics=kinematics,
                    snapshot_path=snap_path
                )

                rec = dict(
                    type=ev['type'],
                    track=tid,
                    t_start=float(ev['t_start']),
                    t_candidate=float(ev['t_candidate']),
                    t_alert=float(t),
                    priority=ev['priority'],
                    event_id=event_id,
                    status="candidate",
                    evidence=evidence
                )
                self.events[event_id] = rec
                new_events.append(rec)

        return new_events


# ----------------------------------------------------------------------------------------------------------------------
# CLI Drawing & Run Pipeline
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]
BANNER = {
    'fall': ('FALL CANDIDATE - operator review', (0, 0, 220)),
    'lying': ('LYING CANDIDATE - operator review', (0, 140, 255)),
    'immobility': ('IMMOBILITY CANDIDATE - operator review', (0, 200, 255))
}


def draw(frame, ids, boxes, kps, t, active, conf):
    for tid, b, k in zip(ids, boxes, kps):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(frame, f'id {int(tid)}', (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        for a, c in COCO_EDGES:
            if k[a, 2] >= conf and k[c, 2] >= conf:
                pt1 = (int(k[a, 0]), int(k[a, 1]))
                pt2 = (int(k[c, 0]), int(k[c, 1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    label, col = 'NORMAL', (0, 140, 0)
    for ty in ('immobility', 'lying', 'fall'):  # fall banner wins
        if ty in active and t - active[ty] <= 5.0:
            label, col = BANNER.get(ty, ('ALERT', (0, 0, 255)))

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), col, -1)
    cv2.putText(frame, f'{label}   t={t:5.1f}s', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def is_live(src):
    return str(src).isdigit() or str(src).lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))


def run(a):
    from ultralytics import YOLO

    cfg = load_cfg(a.config)
    live = is_live(a.source)
    src = int(a.source) if str(a.source).isdigit() else a.source

    fps = 25.0
    if not live:
        cap = cv2.VideoCapture(src)
        f = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = f if f and 1 <= f <= 120 else 25.0

    model = YOLO(a.model)
    evidence_dir = getattr(a, 'evidence_dir', None)
    mon = FallMonitor(cfg, realert_s=a.realert, evidence_dir=evidence_dir)

    kw = dict(
        source=src, stream=True, persist=True, tracker=a.tracker,
        vid_stride=a.stride, imgsz=a.imgsz, conf=a.conf_det, max_det=10, verbose=False
    )
    if a.device is not None:
        kw['device'] = a.device

    sink = open(a.events, 'a', encoding='utf-8') if a.events else None
    writer, active, t0, n_alerts, last_t = None, {}, time.monotonic(), 0, 0.0

    try:
        for n, r in enumerate(model.track(**kw)):
            t = (time.monotonic() - t0) if live else n * a.stride / fps
            last_t = t
            have = r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None and r.keypoints is not None

            ids = r.boxes.id.int().cpu().tolist() if have else []
            boxes = r.boxes.xyxy.cpu().numpy() if have else []
            kps = r.keypoints.data.cpu().numpy() if have else []

            frame_img = r.orig_img.copy() if (a.show or a.save or evidence_dir) and hasattr(r, 'orig_img') else None

            for ev in mon.update(t, ids, boxes, kps, frame=frame_img):
                n_alerts += 1
                active[ev['type']] = t
                print(f"[t={t:6.1f}s] {ev['type'].upper():10s} CANDIDATE  track={ev['track']}  (started ~{ev['t_start']:.1f}s)  priority={ev['priority']}", flush=True)
                if sink:
                    sink.write(json.dumps(ev) + '\n')
                    sink.flush()

            if (a.show or a.save) and frame_img is not None:
                draw(frame_img, ids, boxes, kps, t, active, cfg['sampling']['conf_kp'])
                if a.save and writer is None:
                    writer = cv2.VideoWriter(
                        a.save, cv2.VideoWriter_fourcc(*'mp4v'),
                        (fps / a.stride) if fps else 10.0,
                        (frame_img.shape[1], frame_img.shape[0])
                    )
                if writer:
                    writer.write(frame_img)
                if a.show:
                    cv2.imshow('fall_monitor', frame_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        if sink:
            sink.close()
        if a.show:
            cv2.destroyAllWindows()

    print(f'done: {last_t:.1f}s of video processed, {n_alerts} candidate alert(s)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='video file, webcam index (0), or rtsp/http URL')
    p.add_argument('--model', default='yolo26x-pose.pt', help='Ultralytics pose weights')
    p.add_argument('--tracker', default='bytetrack.yaml', help='bytetrack.yaml or botsort.yaml')
    p.add_argument('--config', help='optional YAML overriding thresholds')
    p.add_argument('--stride', type=int, default=2, help='process every Nth frame')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf-det', type=float, default=0.25, help='person detection confidence')
    p.add_argument('--device', default=None, help="e.g. 0 or 'cpu'")
    p.add_argument('--realert', type=float, default=60.0, help='min seconds between repeat alerts')
    p.add_argument('--show', action='store_true', help='live preview window')
    p.add_argument('--save', help='write an annotated mp4')
    p.add_argument('--events', help='append alerts as JSON lines to this file')
    p.add_argument('--evidence-dir', help='directory where audit snapshots are saved')
    run(p.parse_args())


if __name__ == '__main__':
    main()
