"""T9: transfer_gate.py - YOLO26 -> MMAction2TSSTG transfer evaluation + fine-tune hook (plan Task 3)."""
import numpy as np
import pytest
import torch

import transfer_gate as TG  # ModuleNotFoundError == expected red
from conftest import WEIGHT_DIR, CLASS_NAMES, onehot7
from synth import window_30, IMG_SHAPE


def _seq(kind, label):
    pts, mask = window_30(kind)
    return dict(label=label, pts=pts, conf_mask=mask, img_shape=IMG_SHAPE)


class ScriptedClassifier:
    """Returns the scripted predicted class index per call (order: norm_mode-major, then sequence)."""
    streams = ('j', 'jm')

    def __init__(self, idxs):
        self.idxs = list(idxs)

    def __call__(self, stream_tensors):
        i = self.idxs.pop(0)
        return i, CLASS_NAMES[i], onehot7(i), np.zeros(60)


def test_transfer_gate_evaluates_norm_modes_and_finetune_step(tmp_path):
    seqs = [_seq("fall", "Fall Down"), _seq("sit", "Sit down"), _seq("pickup", "Pickup"), _seq("stand", "Standing")]
    modes = ("prenormalize2d", "track_compact")
    # mode 1 perfect; mode 2 labels the pickup as a fall
    clf = ScriptedClassifier([6, 5, 5, 0] + [6, 5, 6, 0])
    rep = TG.evaluate_tsstg_transfer_on_sequences(seqs, clf, norm_modes=modes)
    a, b = rep["prenormalize2d"], rep["track_compact"]
    assert a["fall_recall"] == 1.0 and a["sitting_specificity"] == 1.0 and a["pickup_specificity"] == 1.0
    assert np.isclose(a["macro_f1"], 1.0)
    assert b["fall_recall"] == 1.0 and b["pickup_specificity"] == 0.0 and b["macro_f1"] < 1.0
    assert rep["recommended_norm_mode"] == "prenormalize2d"
    assert clf.idxs == []                                  # exactly one call per (mode, sequence)

    # fine-tune: loss decreases over 3 steps, adjacency A untouched, result reloadable
    import tsstg_model as M
    base = WEIGHT_DIR / "mmaction2_joint.pth"
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 2, 100, 17, 3)).astype(np.float32)
    y = np.array([42, 7, 5, 8])
    out = tmp_path / "ft.pth"
    stats = TG.finetune_tsstg_on_yolo26(x, y, base, out, epochs=3, lr=1e-3, device="cpu")
    assert stats["loss_end"] < stats["loss_start"]
    assert out.exists()
    A0 = torch.load(base, map_location="cpu", weights_only=True)
    A0 = dict(A0.get("state_dict", A0))["backbone.gcn.0.gcn.A"]
    model = M.load_recognizer_gcn_checkpoint(M.RecognizerGCN(), out)
    assert torch.equal(model.backbone.gcn[0].gcn.A.detach(), A0)


@pytest.mark.real_weights
def test_transfer_gate_real_weights_fall_and_sit():
    import tsstg_model as M
    clf = M.TSSTGClassifier(weight_dir=WEIGHT_DIR, variant="mmaction2", streams=("j", "jm"))
    seqs = [_seq("fall", "Fall Down"), _seq("sit", "Sit down"), _seq("pickup", "Pickup"), _seq("stand", "Standing")]
    rep = TG.evaluate_tsstg_transfer_on_sequences(seqs, clf)
    best = rep[rep["recommended_norm_mode"]]
    assert best["fall_recall"] == 1.0 and best["sitting_specificity"] == 1.0 and best["pickup_specificity"] == 1.0
