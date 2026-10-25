"""evaluate_benchmark.py - Benchmark evaluation, Le2i ground-truth parsing, and metric reporting.

Evaluates human fall, persistent lying, and immobility candidate detection
against standard public benchmarks (Le2i, UR Fall, etc.) or custom video sets.
Computes Precision, Recall, F1-Score, False Alarms per hour (FA/h), Latency,
and activity-specific look-alike breakdown (bending, kneeling, sitting, floor work).
"""
import argparse
import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Suppress harmless optional C++ op notice from mmcv-lite
warnings.filterwarnings("ignore", message=".*MultiScaleDeformableAttention.*")

import cv2
import numpy as np
import torch

try:
    from .fall_monitor import (
        CFG,
        BaselineFallMonitor,
        FallMonitor,
        TSSTGClassifier,
    )
except (ImportError, ValueError):
    from fall_monitor import (
        CFG,
        BaselineFallMonitor,
        FallMonitor,
        TSSTGClassifier,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_benchmark")


# ----------------------------------------------------------------------------------------------------------------------
# Annotation Parsing
# ----------------------------------------------------------------------------------------------------------------------
def parse_le2i_annotation(path: Union[str, Path], fps: float = 25.0) -> Dict[str, Any]:
    """Parse Le2i ground-truth annotation file format.
    
    Standard Le2i format:
      Line 0: fall start frame (int)
      Line 1: fall end frame (int)
      Subsequent lines: bounding box tracking data (frame, id, x1, y1, x2, y2)
      
    If start_frame == 0 and end_frame == 0: normal activity (no fall).
    """
    path = Path(path)
    if not path.exists():
        return {
            "has_fall": False,
            "fall_start_frame": 0,
            "fall_end_frame": 0,
            "fall_start_s": None,
            "fall_end_s": None,
        }

    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [line.strip() for line in raw if line.strip()]
    if not lines:
        return {
            "has_fall": False,
            "fall_start_frame": 0,
            "fall_end_frame": 0,
            "fall_start_s": None,
            "fall_end_s": None,
            "boxes": {},
        }

    fps_val = float(fps) if fps and fps > 0 else 25.0

    # Determine if header has fall frame numbers or if boxes start immediately
    has_header = False
    sf, ef = 0, 0
    box_lines_start = 0

    if len(lines) >= 2 and ("," not in lines[0]) and ("," not in lines[1]):
        try:
            sf = int(lines[0])
            ef = int(lines[1])
            has_header = True
            box_lines_start = 2
        except ValueError:
            has_header = False
            box_lines_start = 0

    has_fall = bool(has_header and (sf > 0 or ef > 0))

    boxes = {}
    for line in lines[box_lines_start:]:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 6:
            try:
                fr = int(parts[0])
                b = np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
                boxes[fr] = b
            except (ValueError, IndexError):
                continue

    return {
        "has_fall": has_fall,
        "fall_start_frame": int(sf),
        "fall_end_frame": int(ef),
        "fall_start_s": float(sf) / fps_val if has_fall else None,
        "fall_end_s": float(ef) / fps_val if has_fall else None,
        "boxes": boxes,
    }


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    b1 = np.asarray(box1, dtype=np.float32)
    b2 = np.asarray(box2, dtype=np.float32)
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(b1[2] - b1[0])) * max(0.0, float(b1[3] - b1[1]))
    area2 = max(0.0, float(b2[2] - b2[0])) * max(0.0, float(b2[3] - b2[1]))
    union_area = area1 + area2 - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def match_fall_event_to_gt(
    event: Dict[str, Any],
    gt: Dict[str, Any],
    tolerance_s: float = 2.0,
    min_iou: float = 0.3
) -> bool:
    """Matches a candidate fall event against ground truth with temporal and optional spatial IoU gate."""
    if not gt.get("has_fall") or gt.get("fall_start_s") is None:
        return False

    gt_start = float(gt["fall_start_s"])
    gt_end = float(gt["fall_end_s"] if gt.get("fall_end_s") is not None else gt_start + 1.0)
    w_min = gt_start - tolerance_s
    w_max = gt_end + tolerance_s

    cand_t = float(event.get("t_candidate", event.get("t_start", 0.0)))
    start_t = float(event.get("t_start", cand_t))

    temporal_match = (w_min <= start_t <= w_max) or (w_min <= cand_t <= w_max)
    if not temporal_match:
        return False

    # Spatial IoU matching if ground-truth bounding box and event box are available
    gt_boxes = gt.get("boxes", {})
    ev_box = event.get("box")
    if ev_box is not None and gt_boxes:
        max_iou = 0.0
        for fr, b in gt_boxes.items():
            max_iou = max(max_iou, compute_box_iou(ev_box, b))
        if max_iou < min_iou:
            return False

    return True


def match_lying_event_to_gt(
    event: Dict[str, Any],
    gt: Dict[str, Any],
    tolerance_s: float = 2.0
) -> bool:
    """Matches a lying candidate event against ground truth lying intervals."""
    if not gt.get("has_lying") or not gt.get("lying_intervals_s"):
        return False

    cand_t = float(event.get("t_candidate", event.get("t_start", 0.0)))
    start_t = float(event.get("t_start", cand_t))

    for s_gt, e_gt in gt["lying_intervals_s"]:
        w_min = float(s_gt) - tolerance_s
        w_max = float(e_gt) + tolerance_s
        if (w_min <= cand_t <= w_max) or (w_min <= start_t <= w_max) or (start_t <= s_gt and cand_t >= e_gt):
            return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
# Metrics Calculation
# ----------------------------------------------------------------------------------------------------------------------
def compute_lying_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics for the persistent lying detector."""
    tp = 0
    fn = 0
    fp = 0
    latencies: List[float] = []

    for r in records:
        gt_lying = bool(r.get("gt_lying", False))
        detected = bool(r.get("detected_lying", False))
        fa = int(r.get("false_alarms_lying", 0))
        fp += fa

        if gt_lying:
            if detected:
                tp += 1
                lat = r.get("latency_lying_s")
                if lat is not None and np.isfinite(lat):
                    latencies.append(float(lat))
            else:
                fn += 1

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_lat = float(np.mean(latencies)) if len(latencies) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_s": mean_lat,
        "total_lying_cases": tp + fn,
    }


def compute_benchmark_metrics(
    records: Sequence[Dict[str, Any]],
    total_duration_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute aggregate benchmark metrics across video evaluation records.
    
    Each record in `records` is expected to have:
      - video: str identifier or path
      - gt_fall: bool (ground truth has fall)
      - detected: bool (true positive fall detected)
      - latency_s: Optional[float] (detection latency in seconds)
      - false_alarms: int (number of false candidate alarms)
      - activity: Optional[str] (e.g. 'bending', 'kneeling', 'sitting', 'floor_work')
    """
    tp = 0
    fn = 0
    fp = 0
    latencies: List[float] = []

    # Initialize standard look-alike breakdown categories
    lookalike_fa: Dict[str, int] = {
        "bending": 0,
        "kneeling": 0,
        "sitting": 0,
        "floor_work": 0,
    }

    for r in records:
        gt_fall = bool(r.get("gt_fall", False))
        detected = bool(r.get("detected", False))
        fa = int(r.get("false_alarms", 0))
        fp += fa

        if gt_fall:
            if detected:
                tp += 1
                lat = r.get("latency_s")
                if lat is not None and np.isfinite(lat):
                    latencies.append(float(lat))
            else:
                fn += 1

        act = r.get("activity")
        if act:
            act_str = str(act)
            lookalike_fa[act_str] = lookalike_fa.get(act_str, 0) + fa

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    dur_h = float(total_duration_hours) if (total_duration_hours is not None and total_duration_hours > 0) else 0.0
    fa_per_hour = float(fp) / dur_h if dur_h > 0 else 0.0
    mean_lat = float(np.mean(latencies)) if len(latencies) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fa_per_hour": fa_per_hour,
        "latency_s": mean_lat,
        "lookalike_fa_breakdown": lookalike_fa,
        "total_duration_hours": dur_h,
        "num_videos": len(records),
        "lying_metrics": compute_lying_metrics(records),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Manifest Parsing
# ----------------------------------------------------------------------------------------------------------------------
def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load video manifest from CSV or JSON file.
    
    Supported formats:
      CSV: video_path,annotation_path,activity (with or without header)
      JSON: list of objects with keys {"video_path", "annotation_path", "activity"}
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = []
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                entries.append({
                    "video_path": str(item.get("video_path", item.get("video", ""))),
                    "annotation_path": str(item.get("annotation_path", item.get("annotation", ""))),
                    "activity": item.get("activity", None),
                })
        return entries

    # CSV parser with header auto-detection
    with manifest_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(c.strip() for c in row)]

    if not rows:
        return []

    first_row = [c.strip().lower() for c in rows[0]]
    has_header = any("video" in c for c in first_row)
    data_rows = rows[1:] if has_header else rows

    for row in data_rows:
        if not row:
            continue
        v_path = row[0].strip() if len(row) > 0 else ""
        a_path = row[1].strip() if len(row) > 1 else ""
        act = row[2].strip() if len(row) > 2 and row[2].strip() else None
        if v_path:
            entries.append({
                "video_path": v_path,
                "annotation_path": a_path,
                "activity": act,
            })

    return entries


# ----------------------------------------------------------------------------------------------------------------------
# Video Evaluation Loop
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_video_manifest(
    manifest_path: Union[str, Path],
    monitor_cls: Optional[Any] = None,
    classifier: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
    tolerance_s: float = 2.0,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    model_name: str = "yolo26x-pose.pt",
    tracker: str = "bytetrack.yaml",
    conf_det: float = 0.25,
    imgsz: int = 640,
    stride: int = 1,
    trace: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute end-to-end benchmark evaluation over videos in manifest."""
    manifest_path = Path(manifest_path)
    entries = load_manifest(manifest_path)
    cfg = cfg or CFG
    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    total_duration_hours = 0.0

    # Instantiate detector and classifier if needed for real videos
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo_model = None
    if any(
        (Path(e["video_path"]).is_absolute() and Path(e["video_path"]).exists())
        or (manifest_path.parent / e["video_path"]).exists()
        or Path(e["video_path"]).exists()
        for e in entries
    ):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model_name)
        except Exception as ex:
            logger.warning("Could not load YOLO model: %s", ex)

    if classifier is None:
        try:
            try:
                from .tsstg_model import TSSTGClassifier
            except (ImportError, ValueError):
                from tsstg_model import TSSTGClassifier
            classifier = TSSTGClassifier(
                weight_dir="weights",
                variant="mmaction2",
                streams=("j", "jm"),
                device=device,
            )
        except Exception as ex:
            logger.warning("Could not load TSSTGClassifier: %s", ex)

    for i, entry in enumerate(entries):
        v_str = entry["video_path"]
        v_path = Path(v_str)
        if not v_path.is_absolute():
            if (manifest_path.parent / v_path).exists():
                v_path = (manifest_path.parent / v_path).resolve()
            elif Path(v_str).exists():
                v_path = Path(v_str).resolve()
            else:
                v_path = (manifest_path.parent / v_path).resolve()

        if not v_path.exists():
            logger.warning("Skipping missing video: %s", v_path)
            continue

        # Inspect video properties
        cap = cv2.VideoCapture(str(v_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 320)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 240)
        cap.release()

        logger.info(
            "Evaluating [%d/%d] %s (%s, %d frames, %.1fs)...",
            i + 1,
            len(entries),
            v_path.name,
            entry.get("activity") or "normal",
            frame_count,
            frame_count / fps if fps > 0 else 0.0,
        )

        duration_s = frame_count / fps if fps > 0 else 0.0
        total_duration_hours += duration_s / 3600.0

        # Parse ground-truth annotation
        ann_str = entry["annotation_path"]
        ann_path = Path(ann_str) if ann_str else None
        if ann_path and not ann_path.is_absolute():
            if (manifest_path.parent / ann_path).exists():
                ann_path = (manifest_path.parent / ann_path).resolve()
            elif Path(ann_str).exists():
                ann_path = Path(ann_str).resolve()
            else:
                ann_path = (manifest_path.parent / ann_path).resolve()

        gt: Dict[str, Any] = {
            "has_fall": False,
            "fall_start_s": None,
            "fall_end_s": None,
            "has_lying": False,
            "lying_intervals_s": [],
        }
        if ann_path and ann_path.exists():
            if ann_path.suffix.lower() == ".txt":
                gt = parse_le2i_annotation(ann_path, fps=fps)
                if gt.get("has_fall") and gt.get("fall_end_s") is not None:
                    gt["has_lying"] = True
                    gt["lying_intervals_s"] = [[gt["fall_end_s"], duration_s]]
                else:
                    gt["has_lying"] = False
                    gt["lying_intervals_s"] = []
            elif ann_path.suffix.lower() == ".json":
                try:
                    gt_data = json.loads(ann_path.read_text(encoding="utf-8"))
                    gt["has_fall"] = bool(gt_data.get("has_fall", False))
                    gt["fall_start_s"] = gt_data.get("fall_start_s")
                    gt["fall_end_s"] = gt_data.get("fall_end_s")
                    gt["has_lying"] = bool(gt_data.get("has_lying", False))
                    gt["lying_intervals_s"] = gt_data.get("lying_intervals_s", [])
                except Exception as ex:
                    logger.warning("Failed parsing JSON annotation %s: %s", ann_path, ex)

        # Run detection and tracking
        active_mon_cls = monitor_cls or FallMonitor
        active_mon_kwargs = dict(kwargs)
        if "img_shape" not in active_mon_kwargs:
            active_mon_kwargs["img_shape"] = (height, width)
        if "device" not in active_mon_kwargs and device:
            active_mon_kwargs["device"] = device
        if "trace" not in active_mon_kwargs:
            active_mon_kwargs["trace"] = trace

        try:
            mon = active_mon_cls(cfg=cfg, classifier=classifier, **active_mon_kwargs)
        except TypeError:
            try:
                mon = active_mon_cls(cfg=cfg, classifier=classifier)
            except TypeError:
                mon = active_mon_cls(cfg=cfg)

        all_events: List[Dict[str, Any]] = []

        if yolo_model is not None:
            track_kwargs = dict(
                source=str(v_path),
                tracker=tracker,
                conf=conf_det,
                imgsz=imgsz,
                vid_stride=stride,
                stream=True,
                verbose=False,
            )
            if device:
                track_kwargs["device"] = device

            idx = 0
            for res in yolo_model.track(**track_kwargs):
                t_curr = float(idx * stride) / fps
                idx += 1
                if res.boxes is None or len(res.boxes) == 0:
                    evs = mon.update(t_curr, [], [], [])
                    all_events.extend(evs)
                    continue

                boxes = res.boxes.xyxy.cpu().numpy()
                ids = (
                    res.boxes.id.cpu().numpy().astype(int)
                    if res.boxes.id is not None
                    else np.arange(len(boxes), dtype=int)
                )
                kps = (
                    res.keypoints.data.cpu().numpy()
                    if res.keypoints is not None
                    else np.zeros((len(boxes), 17, 3), dtype=np.float32)
                )

                evs = mon.update(t_curr, ids, boxes, kps, frame=getattr(res, "orig_img", None))
                all_events.extend(evs)

        # Event matching against ground truth: Falls (All falls + Sustained falls)
        detected_falls = [e for e in all_events if e.get("type") in ("fall", "fall_recovered")]
        detected_sustained = [e for e in all_events if e.get("type") == "fall"]
        has_gt_fall = bool(gt.get("has_fall") and gt.get("fall_start_s") is not None)

        matched_fall = False
        matched_sustained = False
        latency_fall: Optional[float] = None
        false_alarms_fall = 0

        if has_gt_fall:
            gt_start = float(gt["fall_start_s"])
            for ev in detected_falls:
                is_match = match_fall_event_to_gt(ev, gt, tolerance_s=tolerance_s, min_iou=0.3)
                if is_match:
                    if not matched_fall:
                        matched_fall = True
                        cand_t = float(ev.get("t_candidate", ev.get("t_start", 0.0)))
                        alert_t = float(ev.get("t_alert", cand_t))
                        latency_fall = max(0.0, alert_t - gt_start)
                    if ev.get("type") == "fall":
                        matched_sustained = True
                else:
                    if ev.get("type") == "fall":
                        false_alarms_fall += 1
        else:
            false_alarms_fall = len(detected_sustained)

        # Event matching against ground truth: Lying
        detected_lying = [e for e in all_events if e.get("type") == "lying"]
        has_gt_lying = bool(gt.get("has_lying") and gt.get("lying_intervals_s"))

        matched_lying = False
        latency_lying: Optional[float] = None
        false_alarms_lying = 0

        if has_gt_lying:
            for ev in detected_lying:
                is_match = match_lying_event_to_gt(ev, gt, tolerance_s=tolerance_s)
                if is_match:
                    if not matched_lying:
                        matched_lying = True
                        cand_t = float(ev.get("t_candidate", ev.get("t_start", 0.0)))
                        for s_gt, e_gt in gt["lying_intervals_s"]:
                            if float(s_gt) - tolerance_s <= cand_t <= float(e_gt) + tolerance_s:
                                latency_lying = max(0.0, cand_t - float(s_gt))
                                break
                else:
                    false_alarms_lying += 1
        else:
            false_alarms_lying = len(detected_lying)

        clean_events = []
        for e in all_events:
            ev_copy = dict(e)
            if "box" in ev_copy and isinstance(ev_copy["box"], np.ndarray):
                ev_copy["box"] = ev_copy["box"].tolist()
            clean_events.append(ev_copy)

        rec = {
            "video": str(v_path),
            "gt_fall": has_gt_fall,
            "fall_start_s": gt.get("fall_start_s"),
            "fall_end_s": gt.get("fall_end_s"),
            "detected": matched_fall,
            "detected_sustained": matched_sustained,
            "latency_s": latency_fall,
            "false_alarms": false_alarms_fall,
            "gt_lying": has_gt_lying,
            "detected_lying": matched_lying,
            "latency_lying_s": latency_lying,
            "false_alarms_lying": false_alarms_lying,
            "activity": entry.get("activity"),
            "events": clean_events,
        }
        if trace and hasattr(mon, "trace") and mon.trace:
            if (has_gt_fall and not matched_fall) or false_alarms_fall > 0 or false_alarms_lying > 0:
                rec["trace"] = mon.trace
        records.append(rec)

    metrics = compute_benchmark_metrics(records, total_duration_hours=total_duration_hours)

    if out_dir:
        # Save JSON summary
        json_file = out_dir / "benchmark_summary.json"
        summary_payload = {
            "metrics": metrics,
            "records": records,
            "tolerance_s": tolerance_s,
        }
        json_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        logger.info("Saved benchmark metrics to %s", json_file)

        # Save Markdown report
        md_file = out_dir / "benchmark_summary.md"
        md_content = generate_markdown_report(metrics, records, tolerance_s)
        md_file.write_text(md_content, encoding="utf-8")
        logger.info("Saved benchmark markdown report to %s", md_file)

    return metrics


def generate_markdown_report(metrics: Dict[str, Any], records: Sequence[Dict[str, Any]], tolerance_s: float) -> str:
    """Generate publication-ready Markdown summary of benchmark evaluation."""
    lines = [
        "# Benchmark Evaluation Summary Report",
        "",
        f"- **Total Videos Evaluated**: {metrics['num_videos']}",
        f"- **Total Video Duration**: {metrics['total_duration_hours']:.2f} hours",
        f"- **Ground-Truth Matching Tolerance**: $\\pm {tolerance_s:.1f}$ s",
        "",
        "## Fall Detection Performance",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **True Positives (TP)** | {metrics['tp']} |",
        f"| **False Positives (FP)** | {metrics['fp']} |",
        f"| **False Negatives (FN)** | {metrics['fn']} |",
        f"| **Precision** | {metrics['precision']:.4f} |",
        f"| **Recall** | {metrics['recall']:.4f} |",
        f"| **F1-Score** | {metrics['f1']:.4f} |",
        f"| **False Alarms per Hour (FA/h)** | {metrics['fa_per_hour']:.2f} |",
        f"| **Mean Fall Detection Latency** | {metrics['latency_s']:.2f} s |",
        "",
    ]

    lying_m = metrics.get("lying_metrics", {})
    if lying_m and (lying_m.get("total_lying_cases", 0) > 0 or lying_m.get("fp", 0) > 0):
        lines.extend([
            "## Persistent Lying Detection Performance",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **True Positives (TP)** | {lying_m['tp']} |",
            f"| **False Positives (FP)** | {lying_m['fp']} |",
            f"| **False Negatives (FN)** | {lying_m['fn']} |",
            f"| **Precision** | {lying_m['precision']:.4f} |",
            f"| **Recall** | {lying_m['recall']:.4f} |",
            f"| **F1-Score** | {lying_m['f1']:.4f} |",
            f"| **Mean Lying Onset Latency** | {lying_m['latency_s']:.2f} s |",
            "",
        ])

    lines.extend([
        "## Look-Alike False Alarm Breakdown",
        "",
        "| Activity / Posture | False Alarm Count |",
        "| :--- | :--- |",
    ])
    for act, cnt in sorted(metrics.get("lookalike_fa_breakdown", {}).items()):
        lines.append(f"| `{act}` | {cnt} |")

    lines.extend([
        "",
        "## Video Breakdown",
        "",
        "| Video | GT Fall | Det Fall | Fall Lat (s) | GT Lying | Det Lying | Lying Lat (s) | False Alarms | Activity |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in records:
        v_name = Path(r["video"]).name
        gt_fall_txt = "Fall" if r.get("gt_fall") else "-"
        det_fall_txt = "Yes" if r.get("detected") else ("No" if r.get("gt_fall") else "-")
        lat_fall_txt = f"{r['latency_s']:.2f}" if r.get("latency_s") is not None else "-"
        gt_lying_txt = "Lying" if r.get("gt_lying") else "-"
        det_lying_txt = "Yes" if r.get("detected_lying") else ("No" if r.get("gt_lying") else "-")
        lat_lying_txt = f"{r['latency_lying_s']:.2f}" if r.get("latency_lying_s") is not None else "-"
        fa_txt = str(r.get("false_alarms", 0) + r.get("false_alarms_lying", 0))
        act_txt = r.get("activity") or "-"
        lines.append(
            f"| {v_name} | {gt_fall_txt} | {det_fall_txt} | {lat_fall_txt} | "
            f"{gt_lying_txt} | {det_lying_txt} | {lat_lying_txt} | {fa_txt} | {act_txt} |"
        )

    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Human Fall Detection Pipeline on benchmark dataset")
    p.add_argument("--manifest", type=str, required=True, help="Path to CSV/JSON manifest")
    p.add_argument("--model", type=str, default="yolo26x-pose.pt", help="YOLO pose model path")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml", help="ByteTrack YAML config")
    p.add_argument("--tolerance", type=float, default=2.0, help="GT match tolerance in seconds")
    p.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory for output report")
    p.add_argument("--conf-det", type=float, default=0.25, help="Detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    p.add_argument("--stride", type=int, default=1, help="Video frame evaluation stride")
    p.add_argument("--device", type=str, default=None, help="Inference device (cpu, cuda:0)")
    p.add_argument("--trace", action="store_true", help="Log diagnostic kinematic trace for error analysis")
    return p


def main():
    parser = build_cli_parser()
    args = parser.parse_args()
    evaluate_video_manifest(
        manifest_path=args.manifest,
        model_name=args.model,
        tracker=args.tracker,
        tolerance_s=args.tolerance,
        output_dir=args.output_dir,
        conf_det=args.conf_det,
        imgsz=args.imgsz,
        stride=args.stride,
        device=args.device,
        trace=args.trace,
    )


if __name__ == "__main__":
    main()
