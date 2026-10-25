"""tsstg_model.py - Standalone OpenMMLab MMAction2 RecognizerGCN & Multi-Stream Classifier.

Wraps upstream OpenMMLab MMAction2 STGCN backbone, Graph, GCNHead, with checkpoint verification,
weight remapping for MMAction2/PYSKL checkpoints, multi-stream fusion, and NTU-60 to 7-class projection.
"""
from pathlib import Path
from typing import Sequence, Dict, Tuple, Any, Optional, Union
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Suppress harmless optional C++ op notice from mmcv-lite when imported via mmaction2
warnings.filterwarnings("ignore", message=".*MultiScaleDeformableAttention.*")

from mmaction.models.utils.graph import Graph, normalize_digraph
from mmaction.models.backbones.stgcn import STGCN
from mmaction.models.heads.gcn_head import GCNHead

CLASS_NAMES: Tuple[str, ...] = (
    'Standing',
    'Walking',
    'Sitting',
    'Lying Down',
    'Stand up',
    'Sit down',
    'Fall Down'
)

CLASS_NAMES_8: Tuple[str, ...] = (
    'Standing',
    'Walking',
    'Sitting',
    'Lying Down',
    'Stand up',
    'Sit down',
    'Fall Down',
    'Instability'
)

class ActionClass:
    STANDING = "Standing"
    WALKING = "Walking"
    SITTING = "Sitting"
    LYING = "Lying Down"
    STAND_UP = "Stand up"
    SIT_DOWN = "Sit down"
    FALL = "Fall Down"
    INSTABILITY = "Instability"

CLASS_TO_IDX_7 = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX_8 = {name: i for i, name in enumerate(CLASS_NAMES_8)}

NTU60_CLASS_NAMES: Tuple[str, ...] = (
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up (from sitting position)",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other"
)

WEIGHT_SHA256_REGISTRY: Dict[str, str] = {
    "mmaction2_joint.pth": "484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea",
    "mmaction2_bone.pth": "c4b44488123d68e19d742ade26d02ff638a911bb0084fcfa7b54d751957318dd",
    "mmaction2_jm.pth": "f18eb408a42eebced02fc0e0dfeed2c6edb320cc1752abc6df855dff0845fc6d",
    "mmaction2_bm.pth": "99c60e2d462a3978114193f35b5fce465bfa32de2275325913ab8a394900bfb0",
    "pyskl_j.pth": "b4e476a34f2fe641f111cf8b8b66d08fca163a0b45e3df6b9ebb5c93ac9a71b2",
    "pyskl_b.pth": "e9d7cae55964837e6bd9ba5b00f346655da1937317421d79a8963fa9cf0c7fed",
    "pyskl_jm.pth": "131738564797c3d1c4fc5f3580e4cda0303cffa16efb5f46f4c924e6d9488615",
    "pyskl_bm.pth": "0e55b74c7d50410ec64ddb4527be6c83145ced62ed671383df45ccd1a2225ac4",
}


def verify_sha256(path: Union[str, Path], expected_sha256: Optional[str] = None) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"SHA-256 verification failed for {path.name}: expected {expected_sha256}, got {digest}"
        )
    return digest


class RecognizerGCN(nn.Module):
    """Inference-oriented RecognizerGCN with STGCN backbone and GCNHead."""
    def __init__(self, num_classes: int = 60, in_channels: int = 3, graph_cfg: Optional[Dict] = None):
        super().__init__()
        if graph_cfg is None:
            graph_cfg = dict(layout='coco', mode='stgcn_spatial')
        self.backbone = STGCN(graph_cfg=graph_cfg, in_channels=in_channels)
        self.cls_head = GCNHead(num_classes=num_classes, in_channels=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.cls_head(feat)


def load_recognizer_gcn_checkpoint(
    model: RecognizerGCN,
    weight_path: Union[str, Path],
    device: str = "cpu",
    expected_sha256: Optional[str] = None
) -> RecognizerGCN:
    weight_path = Path(weight_path)
    if expected_sha256 is None:
        expected_sha256 = WEIGHT_SHA256_REGISTRY.get(weight_path.name)
    verify_sha256(weight_path, expected_sha256=expected_sha256)

    ckpt = torch.load(weight_path, map_location=device, weights_only=True)
    sd = dict(ckpt.get("state_dict", ckpt))

    remapped_sd = {}
    for k, v in sd.items():
        if k.startswith("cls_head.fc_cls."):
            remapped_sd[k.replace("cls_head.fc_cls.", "cls_head.fc.")] = v
        else:
            remapped_sd[k] = v

    if "backbone.gcn.0.gcn.A" in remapped_sd:
        built_A = model.backbone.gcn[0].gcn.A
        ckpt_A = remapped_sd["backbone.gcn.0.gcn.A"].to(built_A.device)
        if not torch.allclose(built_A, ckpt_A, atol=1e-6):
            raise ValueError(f"Adjacency matrix buffer A does not match checkpoint {weight_path.name}")

    model.load_state_dict(remapped_sd, strict=True)
    model.to(device).eval()
    return model


def load_action_projection_cfg(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else (ROOT / "config.yaml" if (ROOT / "config.yaml").exists() else Path("config.yaml"))
    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if "action_classes" in data and "projection" in data["action_classes"]:
                return data["action_classes"]["projection"]
        except Exception:
            pass
    return {
        "fall_sources": [42],
        "instability_sources": [41],
        "sitdown_sources": [5, 7, 15, 16],
        "standup_sources": [8],
        "walking_sources": [25, 26, 58, 59],
        "sitting_sources": [10, 11, 28, 29]
    }


def project_ntu60_to_8_classes(
    ntu60_probs: Union[np.ndarray, torch.Tensor],
    projection_cfg: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Project 60-class NTU-60 probabilities onto the 8-class pipeline action taxonomy.

    Classes:
      0: Standing    (residual mass)
      1: Walking     (A26 hopping, A27 jump up, A59 walking towards, A60 walking apart) -- A42 EXCLUDED!
      2: Sitting     (A11 reading, A12 writing, A29 phone/tablet, A30 typing)
      3: Lying Down  (0.0, handled in kinematic state machine)
      4: Stand up    (A9, idx 8)
      5: Sit down    (A6, A8, A16, A17)
      6: Fall Down   (A43, idx 42)
      7: Instability (A42, idx 41: staggering) -- used strictly for Path B kinematic support
    """
    if projection_cfg is None:
        projection_cfg = load_action_projection_cfg()

    if isinstance(ntu60_probs, torch.Tensor):
        p = ntu60_probs.detach().cpu().numpy()
    else:
        p = np.asarray(ntu60_probs, dtype=np.float64)

    single = (p.ndim == 1)
    if single:
        p = p[None, :]

    q = np.zeros((p.shape[0], 8), dtype=np.float64)

    def _map_src(src_list):
        if not src_list:
            return 0.0
        return p[:, src_list].sum(axis=1)

    # class 6: Fall Down (A43 default idx 42)
    q[:, 6] = _map_src(projection_cfg.get("fall_sources", [42]))

    # class 7: Instability (A42 default idx 41)
    q[:, 7] = _map_src(projection_cfg.get("instability_sources", [41]))

    # class 5: Sit down / controlled bend & kneel (A6, A8, A16, A17)
    q[:, 5] = _map_src(projection_cfg.get("sitdown_sources", [5, 7, 15, 16]))

    # class 4: Stand up (A9 default idx 8)
    q[:, 4] = _map_src(projection_cfg.get("standup_sources", [8]))

    # class 1: Walking (A26 hopping, A27 jump up, A59 walking towards, A60 walking apart)
    q[:, 1] = _map_src(projection_cfg.get("walking_sources", [25, 26, 58, 59]))

    # class 2: Sitting (A11 reading, A12 writing, A29 phone/tablet, A30 typing)
    q[:, 2] = _map_src(projection_cfg.get("sitting_sources", [10, 11, 28, 29]))

    # class 3: Lying Down (always 0 in raw NTU-60 head; handled in state machine)
    q[:, 3] = 0.0

    # class 0: Standing (residual mass)
    mapped_sum = q[:, 1:].sum(axis=1)
    q[:, 0] = np.maximum(0.0, 1.0 - mapped_sum)

    return q[0] if single else q


def project_ntu60_to_7_classes(ntu60_probs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(ntu60_probs, torch.Tensor):
        p = ntu60_probs.detach().cpu().numpy()
    else:
        p = np.asarray(ntu60_probs, dtype=np.float64)

    single = (p.ndim == 1)
    if single:
        p = p[None, :]

    q = np.zeros((p.shape[0], 7), dtype=np.float64)
    # class 6: Fall Down (A43, idx 42)
    q[:, 6] = p[:, 42]
    # class 5: Sit down / controlled bend & kneel (A6, A8, A16, A17)
    q[:, 5] = p[:, [5, 7, 15, 16]].sum(axis=1)
    # class 4: Stand up (A9, idx 8)
    q[:, 4] = p[:, 8]
    # class 1: Walking (A26 hopping, A27 jump up, A42 staggering, A59 walking towards, A60 walking apart)
    q[:, 1] = p[:, [25, 26, 41, 58, 59]].sum(axis=1)
    # class 2: Sitting (A11 reading, A12 writing, A29 phone/tablet, A30 typing)
    q[:, 2] = p[:, [10, 11, 28, 29]].sum(axis=1)
    # class 3: Lying Down (always 0 in raw NTU-60 head; handled in state machine)
    q[:, 3] = 0.0
    # class 0: Standing (residual mass)
    mapped_sum = q[:, 1:].sum(axis=1)
    q[:, 0] = np.maximum(0.0, 1.0 - mapped_sum)

    return q[0] if single else q


STREAM_FILE_MAP = {
    'mmaction2': {
        'j': 'mmaction2_joint.pth',
        'jm': 'mmaction2_jm.pth',
        'b': 'mmaction2_bone.pth',
        'bm': 'mmaction2_bm.pth',
    },
    'pyskl': {
        'j': 'pyskl_j.pth',
        'jm': 'pyskl_jm.pth',
        'b': 'pyskl_b.pth',
        'bm': 'pyskl_bm.pth',
    }
}


class TSSTGClassifier:
    """Multi-stream STGCN action classifier loading OpenMMLab / PYSKL checkpoints."""
    def __init__(
        self,
        weight_dir: Union[str, Path] = "weights",
        variant: str = "mmaction2",
        streams: Sequence[str] = ("j", "jm"),
        stream_weights: Optional[Sequence[float]] = None,
        num_classes: int = 8,
        device: str = "cpu",
        projection_cfg: Optional[Dict[str, Any]] = None
    ):
        self.weight_dir = Path(weight_dir)
        self.variant = variant.lower()
        self.streams = tuple(streams)
        self.device = device
        self.num_classes = num_classes
        self.projection_cfg = projection_cfg or load_action_projection_cfg()

        if stream_weights is None:
            self.stream_weights = [1.0 / len(self.streams)] * len(self.streams)
        else:
            total = sum(stream_weights)
            self.stream_weights = [w / total for w in stream_weights]

        if self.variant not in STREAM_FILE_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. Valid options: {list(STREAM_FILE_MAP)}"
            )
        file_map = STREAM_FILE_MAP[self.variant]
        self.models: Dict[str, RecognizerGCN] = {}
        for s in self.streams:
            fname = file_map.get(s)
            if fname is None:
                raise ValueError(f"Unknown stream '{s}' for variant '{variant}'")
            fpath = self.weight_dir / fname
            model = RecognizerGCN()
            load_recognizer_gcn_checkpoint(model, fpath, device=self.device)
            self.models[s] = model

    def __call__(
        self,
        stream_tensors: Dict[str, torch.Tensor]
    ) -> Tuple[int, str, np.ndarray, np.ndarray]:
        ntu_probs_total: Optional[np.ndarray] = None

        with torch.no_grad():
            for s, weight in zip(self.streams, self.stream_weights):
                x = stream_tensors[s].to(self.device)
                logits = self.models[s](x)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float64)
                if ntu_probs_total is None:
                    ntu_probs_total = weight * probs
                else:
                    ntu_probs_total += weight * probs

        assert ntu_probs_total is not None
        if self.num_classes == 8:
            probs = project_ntu60_to_8_classes(ntu_probs_total, projection_cfg=self.projection_cfg)
            pred_idx = int(np.argmax(probs))
            class_name = CLASS_NAMES_8[pred_idx]
        else:
            probs = project_ntu60_to_7_classes(ntu_probs_total)
            pred_idx = int(np.argmax(probs))
            class_name = CLASS_NAMES[pred_idx]

        return pred_idx, class_name, probs, ntu_probs_total
