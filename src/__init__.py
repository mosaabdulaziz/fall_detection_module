"""fall_detection - Core Human Fall & Persistent Lying Detection System."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .fall_monitor import (
    FallMonitor,
    BaselineFallMonitor,
    load_cfg,
    compute_features,
    detect_falls,
    detect_lying,
    THRESHOLDS,
    CFG,
)
from .keypoint_adapter import (
    sanitize_keypoints_17,
    resample_window_25hz,
    build_mmaction2_tensors,
    prenormalize_2d,
    gen_ske_feat,
    uniform_sample_and_format_gcn,
)
# Alias for backwards-compatibility
prenormalize2d_coco_17 = prenormalize_2d

from .tsstg_model import (
    TSSTGClassifier,
    RecognizerGCN,
    load_recognizer_gcn_checkpoint,
    CLASS_NAMES,
    CLASS_NAMES_8,
    ActionClass,
    CLASS_TO_IDX_7,
    CLASS_TO_IDX_8,
    NTU60_CLASS_NAMES,
    project_ntu60_to_7_classes,
    project_ntu60_to_8_classes,
)
from .transfer_gate import (
    evaluate_tsstg_transfer_on_sequences,
    finetune_tsstg_on_yolo26,
)
from .evaluate_benchmark import (
    evaluate_video_manifest,
    parse_le2i_annotation,
    compute_benchmark_metrics,
    compute_lying_metrics,
    match_fall_event_to_gt,
    match_lying_event_to_gt,
)

__all__ = [
    "FallMonitor",
    "BaselineFallMonitor",
    "load_cfg",
    "compute_features",
    "detect_falls",
    "detect_lying",
    "THRESHOLDS",
    "CFG",
    "sanitize_keypoints_17",
    "resample_window_25hz",
    "build_mmaction2_tensors",
    "prenormalize_2d",
    "prenormalize2d_coco_17",
    "gen_ske_feat",
    "uniform_sample_and_format_gcn",
    "TSSTGClassifier",
    "RecognizerGCN",
    "load_recognizer_gcn_checkpoint",
    "CLASS_NAMES",
    "NTU60_CLASS_NAMES",
    "project_ntu60_to_7_classes",
    "evaluate_tsstg_transfer_on_sequences",
    "finetune_tsstg_on_yolo26",
    "evaluate_video_manifest",
    "parse_le2i_annotation",
    "compute_benchmark_metrics",
    "compute_lying_metrics",
    "match_fall_event_to_gt",
    "match_lying_event_to_gt",
]
