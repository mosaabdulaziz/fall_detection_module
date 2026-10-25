import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WEIGHT_DIR = ROOT / "weights"
CLASS_NAMES = ('Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down')
STAND_, WALK_, SITTING_, LYING_, STANDUP_, SITDOWN_, FALL_ = range(7)


def onehot7(idx, mass=0.9):
    p = np.full(7, (1.0 - mass) / 6.0, dtype=np.float64)
    p[idx] = mass
    return p


class FakeClassifier:
    """Stands in for TSSTGClassifier: callable(stream_tensors) -> (pred_idx_7, name_7, probs_7, ntu60_probs)."""
    streams = ('j', 'jm')

    def __init__(self, probs7=None):
        self.probs7 = onehot7(STAND_) if probs7 is None else np.asarray(probs7, dtype=np.float64)
        self.calls = 0

    def __call__(self, stream_tensors):
        self.calls += 1
        for name, x in stream_tensors.items():
            assert tuple(x.shape) == (1, 2, 100, 17, 3), (name, tuple(x.shape))
        p = self.probs7
        ntu = np.zeros(60); ntu[42] = p[FALL_]; ntu[7] = p[SITDOWN_]; ntu[10] = p[SITTING_]
        i = int(p.argmax())
        return i, CLASS_NAMES[i], p, ntu


@pytest.fixture
def fake_cls():
    return FakeClassifier()


def drive(mon, fake, t, kps, boxes, probs_fn=None, tid=1, frames=None):
    """Feed one track through FallMonitor.update and return every emitted event."""
    events = []
    for i in range(len(t)):
        if probs_fn is not None:
            fake.probs7 = probs_fn(float(t[i]))
        fr = None if frames is None else frames[i]
        events += mon.update(float(t[i]), [tid], [boxes[i]], [kps[i]], frame=fr) if fr is not None \
            else mon.update(float(t[i]), [tid], [boxes[i]], [kps[i]])
    return events
