"""Deterministic synthetic COCO-17 skeleton tracks used by the test-suite (no video / no detector needed).

Pixel space: 1280x720 frame, floor at y=700, a standing person is H=300 px tall.
COCO order: 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelb,8 relb,9 lwri,10 rwri,
            11 lhip,12 rhip,13 lknee,14 rknee,15 lank,16 rank
"""
import numpy as np

H = 300.0
FLOOR = 700.0
IMG_SHAPE = (720, 1280)  # (h, w)

# pose parameters: hx, hy = hip centre; a = torso angle from vertical (deg); s = 0 straight legs .. 1 seated legs
STAND = dict(hx=640.0, hy=FLOOR - 0.50 * H, a=0.0, s=0.0)
SIT = dict(hx=640.0, hy=FLOOR - 0.25 * H, a=0.0, s=1.0)
LIE = dict(hx=640.0, hy=FLOOR - 0.07 * H, a=90.0, s=0.0)


def make_pose(hx, hy, a, s=0.0, conf=0.9):
    r = np.radians(a)
    u = np.array([np.sin(r), -np.cos(r)])   # body "up" axis
    p = np.array([np.cos(r), np.sin(r)])    # lateral axis
    hip = np.array([hx, hy])
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = conf

    def put(i, pt):
        k[i, :2] = pt
    put(11, hip - 0.06 * H * p); put(12, hip + 0.06 * H * p)
    sh = hip + 0.30 * H * u
    put(5, sh - 0.10 * H * p); put(6, sh + 0.10 * H * p)
    head = hip + 0.42 * H * u
    put(0, head); put(1, head + 0.02 * H * u - 0.02 * H * p); put(2, head + 0.02 * H * u + 0.02 * H * p)
    put(3, head - 0.05 * H * p); put(4, head + 0.05 * H * p)
    put(7, sh - 0.15 * H * u - 0.10 * H * p); put(8, sh - 0.15 * H * u + 0.10 * H * p)
    put(9, sh - 0.28 * H * u - 0.10 * H * p); put(10, sh - 0.28 * H * u + 0.10 * H * p)
    for kn, an, side in ((13, 15, -1), (14, 16, 1)):
        base = hip + side * 0.06 * H * p
        k_st = base - 0.25 * H * u
        a_st = base - 0.50 * H * u
        k_sit = base + np.array([0.25 * H, 0.0])
        a_sit = k_sit + np.array([0.0, 0.25 * H])
        put(kn, (1 - s) * k_st + s * k_sit)
        put(an, (1 - s) * a_st + s * a_sit)
    return k


def box_of(kp, margin=0.08):
    x0, y0 = kp[:, :2].min(0); x1, y1 = kp[:, :2].max(0)
    w, h = x1 - x0, y1 - y0
    return np.array([x0 - margin * w, y0 - margin * h, x1 + margin * w, y1 + margin * h], dtype=np.float32)


def track(segments, fps=25.0, t0=0.0):
    """segments: list of (duration_s, params_start, params_end) linearly interpolated.
    Returns t (N,), kps (N,17,3), boxes (N,4)."""
    ts, ks, bs = [], [], []
    t = t0
    for dur, p0, p1 in segments:
        n = int(round(dur * fps))
        for i in range(n):
            f = i / max(n, 1)
            prm = {key: (1 - f) * p0[key] + f * p1[key] for key in p0}
            kp = make_pose(**prm)
            ts.append(t); ks.append(kp); bs.append(box_of(kp)); t += 1.0 / fps
    return np.array(ts), np.array(ks), np.array(bs)


def hold(params, dur):
    return (dur, params, params)


def jitter(kps, amp=1.5, seed=0):
    rng = np.random.default_rng(seed)
    out = kps.copy()
    out[..., :2] += rng.uniform(-amp, amp, size=out[..., :2].shape).astype(np.float32)
    return out


def window_30(kind, conf_mask=True):
    """A 1.16 s / 25 Hz / 30-frame window (30,17,3) in pixels + (30,17) confidence mask."""
    if kind == "stand":
        segs = [hold(STAND, 1.2)]
    elif kind == "walk":
        segs = [(1.2, dict(STAND, hx=400.0), dict(STAND, hx=700.0))]
    elif kind == "fall":
        segs = [hold(STAND, 0.4), (0.3, STAND, LIE), hold(LIE, 0.5)]
    elif kind == "sit":
        segs = [hold(STAND, 0.3), (0.5, STAND, SIT), hold(SIT, 0.4)]
    elif kind == "pickup":
        segs = [hold(STAND, 0.3), (0.4, STAND, dict(STAND, hy=FLOOR - 0.38 * H, a=70.0)),
                (0.4, dict(STAND, hy=FLOOR - 0.38 * H, a=70.0), STAND)]
    else:
        raise KeyError(kind)
    _, ks, _ = track(segs, fps=25.0)
    ks = ks[-30:]
    return ks, np.ones((30, 17), dtype=bool)
