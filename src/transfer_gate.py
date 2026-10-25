"""transfer_gate.py - YOLO26 to MMAction2TSSTG Transfer Evaluation & Fine-Tuning Hook.

Provides:
1. evaluate_tsstg_transfer_on_sequences: Evaluates action recognition transfer across
   different coordinate normalization modes ('prenormalize2d' vs 'track_compact') on key
   action categories and controlled look-alikes (sitting down, picking up, falling).
2. finetune_tsstg_on_yolo26: Fine-tuning routine to adapt RecognizerGCN onto extracted
   skeletons while strictly preserving the graph topology buffer A.
"""
from pathlib import Path
from typing import Sequence, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    from .keypoint_adapter import build_mmaction2_tensors
    from .tsstg_model import (
        RecognizerGCN,
        load_recognizer_gcn_checkpoint,
        TSSTGClassifier
    )
except (ImportError, ValueError):
    from keypoint_adapter import build_mmaction2_tensors
    from tsstg_model import (
        RecognizerGCN,
        load_recognizer_gcn_checkpoint,
        TSSTGClassifier
    )

LABEL_TO_CLASS7: Dict[str, int] = {
    "fall down": 6,
    "fall": 6,
    "sit down": 5,
    "sit": 5,
    "pickup": 5,
    "pick up": 5,
    "stand up": 4,
    "lying down": 3,
    "lying": 3,
    "sitting": 2,
    "walking": 1,
    "walk": 1,
    "standing": 0,
    "stand": 0
}


def _compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    classes = sorted(list(set(y_true) | set(y_pred)))
    if not classes:
        return 0.0

    f1_scores = []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        denom = 2 * tp + fp + fn
        if denom == 0:
            f1 = 1.0 if (tp + fp + fn == 0) else 0.0
        else:
            f1 = (2.0 * tp) / denom
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def evaluate_tsstg_transfer_on_sequences(
    sequences: Sequence[Dict[str, Any]],
    classifier: TSSTGClassifier,
    norm_modes: Sequence[str] = ("prenormalize2d", "track_compact")
) -> Dict[str, Any]:
    """Evaluates classifier transfer on labeled skeleton sequences across normalization modes."""
    report: Dict[str, Any] = {}
    best_macro_f1 = -1.0
    recommended_mode = norm_modes[0]

    for mode in norm_modes:
        y_true = []
        y_pred = []
        raw_labels = []

        for seq in sequences:
            lbl = seq["label"].strip().lower()
            target_class = LABEL_TO_CLASS7.get(lbl, 0)
            y_true.append(target_class)
            raw_labels.append(lbl)

            img_shape = seq.get("img_shape", (720, 1280))
            tensors = build_mmaction2_tensors(
                seq["pts"],
                conf_mask_30_17=seq.get("conf_mask"),
                img_shape=img_shape,
                streams=classifier.streams,
                norm_mode=mode
            )
            pred_idx, _, _, _ = classifier(tensors)
            y_pred.append(pred_idx)

        # Fall Recall: True falls correctly identified as Fall Down (idx 6)
        fall_indices = [i for i, lbl in enumerate(raw_labels) if "fall" in lbl]
        if fall_indices:
            fall_recall = sum(1 for i in fall_indices if y_pred[i] == 6) / len(fall_indices)
        else:
            fall_recall = 1.0

        # Sitting Specificity: Sit-down actions NOT falsely predicted as Fall Down (idx 6)
        sit_indices = [i for i, lbl in enumerate(raw_labels) if "sit" in lbl]
        if sit_indices:
            sitting_specificity = sum(1 for i in sit_indices if y_pred[i] != 6) / len(sit_indices)
        else:
            sitting_specificity = 1.0

        # Pickup Specificity: Bending/Pickup actions NOT falsely predicted as Fall Down (idx 6)
        pickup_indices = [i for i, lbl in enumerate(raw_labels) if "pick" in lbl]
        if pickup_indices:
            pickup_specificity = sum(1 for i in pickup_indices if y_pred[i] != 6) / len(pickup_indices)
        else:
            pickup_specificity = 1.0

        macro_f1 = _compute_macro_f1(y_true, y_pred)

        report[mode] = {
            "fall_recall": float(fall_recall),
            "sitting_specificity": float(sitting_specificity),
            "pickup_specificity": float(pickup_specificity),
            "macro_f1": float(macro_f1)
        }

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            recommended_mode = mode

    report["recommended_norm_mode"] = recommended_mode
    return report


def finetune_tsstg_on_yolo26(
    train_windows: Union[np.ndarray, torch.Tensor],
    train_labels_ntu60: Union[np.ndarray, torch.Tensor],
    base_weight_path: Union[str, Path],
    out_weight_path: Union[str, Path],
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, float]:
    """Fine-tuning hook for adapting RecognizerGCN onto YOLO26 extracted skeletons."""
    out_weight_path = Path(out_weight_path)
    out_weight_path.parent.mkdir(parents=True, exist_ok=True)

    model = RecognizerGCN()
    load_recognizer_gcn_checkpoint(model, base_weight_path, device=device)

    # Initial graph buffer state
    initial_A = model.backbone.gcn[0].gcn.A.clone().detach()

    x = torch.as_tensor(train_windows, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_labels_ntu60, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        initial_logits = model(x)
        loss_start = float(criterion(initial_logits, y).item())

    # Keep BatchNorm in eval mode to prevent running stats corruption on small transfer batches
    model.eval()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = model(x)
        loss_end = float(criterion(final_logits, y).item())

    # Verify adjacency buffer A was strictly preserved
    if not torch.equal(model.backbone.gcn[0].gcn.A.detach(), initial_A):
        raise RuntimeError("Graph adjacency matrix buffer A was mutated during fine-tuning!")

    torch.save({"state_dict": model.state_dict()}, out_weight_path)

    return {
        "loss_start": loss_start,
        "loss_end": loss_end
    }
