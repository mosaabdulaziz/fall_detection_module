"""T1-T4: tsstg_model.py - OpenMMLab ST-GCN (17 COCO kpts), checkpoints, 60->7 projection (plan Task 1)."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import WEIGHT_DIR, CLASS_NAMES

import tsstg_model as M  # noqa: E402  (ModuleNotFoundError == expected red)

CKPTS_ALL = {
    "mmaction2_joint.pth": "484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea",
    "mmaction2_bone.pth": "c4b44488123d68e19d742ade26d02ff638a911bb0084fcfa7b54d751957318dd",
    "mmaction2_jm.pth": "f18eb408a42eebced02fc0e0dfeed2c6edb320cc1752abc6df855dff0845fc6d",
    "mmaction2_bm.pth": "99c60e2d462a3978114193f35b5fce465bfa32de2275325913ab8a394900bfb0",
    "pyskl_j.pth": "b4e476a34f2fe641f111cf8b8b66d08fca163a0b45e3df6b9ebb5c93ac9a71b2",
    "pyskl_b.pth": "e9d7cae55964837e6bd9ba5b00f346655da1937317421d79a8963fa9cf0c7fed",
    "pyskl_jm.pth": "131738564797c3d1c4fc5f3580e4cda0303cffa16efb5f46f4c924e6d9488615",
    "pyskl_bm.pth": "0e55b74c7d50410ec64ddb4527be6c83145ced62ed671383df45ccd1a2225ac4",
}


def _sd(name):
    ck = torch.load(WEIGHT_DIR / name, map_location="cpu", weights_only=True)
    return dict(ck.get("state_dict", ck))


# ---------------------------------------------------------------- T1
def test_mmaction2_stgcn_ten_stages_matches_code_divergence():
    model = M.RecognizerGCN()
    blocks = model.backbone.gcn
    assert len(blocks) == 10                                    # paper says 9, code says 10 (E111a / E203)
    assert blocks[0].residual(torch.zeros(1)) == 0              # stage 0 has no residual
    assert [b.tcn.conv.out_channels for b in blocks] == [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
    strides = [b.tcn.conv.stride[0] for b in blocks]
    assert strides[4] == 2 and strides[7] == 2
    assert sum(s == 2 for s in strides) == 2
    assert model.backbone.data_bn.num_features == 51
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 2, 100, 17, 3))
    assert tuple(out.shape) == (2, 60) and torch.isfinite(out).all()


# ---------------------------------------------------------------- T2
def test_coco_17_graph_matches_checkpoint_buffer_before_load():
    # normalize_digraph == A @ D^-1 (column normalisation), not the paper's symmetric form
    A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1.0]])
    expected = A @ np.diag(1.0 / A.sum(0))
    np.testing.assert_allclose(M.normalize_digraph(A), expected, atol=1e-12)

    g = M.Graph(layout="coco", mode="stgcn_spatial")
    assert g.num_node == 17 and g.A.shape == (3, 17, 17)
    model = M.RecognizerGCN()
    built = model.backbone.gcn[0].gcn.A.detach()
    assert torch.allclose(built, torch.tensor(g.A, dtype=torch.float32), atol=1e-6)
    for name in ("mmaction2_joint.pth", "mmaction2_jm.pth", "pyskl_j.pth", "pyskl_b.pth"):
        sd = _sd(name)
        assert torch.allclose(built, sd["backbone.gcn.0.gcn.A"], atol=1e-6), name


# ---------------------------------------------------------------- T3
@pytest.mark.parametrize("name", ["mmaction2_joint.pth", "mmaction2_jm.pth", "mmaction2_bone.pth",
                                  "pyskl_j.pth", "pyskl_jm.pth"])
def test_mmaction2_and_pyskl_checkpoints_load_strictly_with_sha256(name):
    path = WEIGHT_DIR / name
    assert M.WEIGHT_SHA256_REGISTRY[name] == CKPTS_ALL[name]
    assert M.verify_sha256(path) == CKPTS_ALL[name]
    model = M.load_recognizer_gcn_checkpoint(M.RecognizerGCN(), path)   # strict=True inside
    assert not model.training
    assert hasattr(model.cls_head, "fc") and not hasattr(model.cls_head, "fc_cls")


def test_tampered_checkpoint_is_rejected(tmp_path):
    src = WEIGHT_DIR / "mmaction2_joint.pth"
    bad = tmp_path / "mmaction2_joint.pth"
    shutil.copy(src, bad)
    data = bytearray(bad.read_bytes()); data[len(data) // 2] ^= 0xFF; bad.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        M.verify_sha256(bad, CKPTS_ALL["mmaction2_joint.pth"])
    with pytest.raises(ValueError):
        M.load_recognizer_gcn_checkpoint(M.RecognizerGCN(), bad, expected_sha256=CKPTS_ALL["mmaction2_joint.pth"])


def test_projection_ntu60_to_7_classes_is_a_distribution():
    p = np.zeros(60); p[42] = 0.2; p[7] = 0.1; p[5] = 0.1; p[8] = 0.05; p[26] = 0.1; p[11] = 0.05; p[0] = 0.4
    q = M.project_ntu60_to_7_classes(p)
    assert q.shape == (7,) and np.isclose(q.sum(), 1.0)
    assert np.isclose(q[6], 0.2) and np.isclose(q[5], 0.2) and np.isclose(q[4], 0.05)
    assert np.isclose(q[1], 0.1) and np.isclose(q[2], 0.05) and q[3] == 0.0
    assert np.isclose(q[0], 1.0 - 0.2 - 0.2 - 0.05 - 0.1 - 0.05)
    assert M.CLASS_NAMES == CLASS_NAMES and len(M.NTU60_CLASS_NAMES) == 60
    assert M.NTU60_CLASS_NAMES[42] == "falling" and M.NTU60_CLASS_NAMES[7] == "sitting down"


# ---------------------------------------------------------------- T4
@pytest.mark.real_weights
def test_mmaction2_tsstg_discriminates_fall_sitdown_and_pickup_on_real_weights():
    from keypoint_adapter import build_mmaction2_tensors
    from synth import window_30, IMG_SHAPE
    clf = M.TSSTGClassifier(weight_dir=WEIGHT_DIR, variant="mmaction2", streams=("j", "jm"))

    def run(kind):
        pts, mask = window_30(kind)
        return clf(build_mmaction2_tensors(pts, mask, img_shape=IMG_SHAPE, streams=("j", "jm")))

    idx, name, p7, ntu = run("fall")
    assert idx == 6 and p7[6] > 0.50
    idx, name, p7, ntu = run("sit")
    assert idx == 5 and p7[6] < 0.20 and ntu[7] > 0.80
    idx, name, p7, ntu = run("pickup")
    assert p7[6] < 0.10
