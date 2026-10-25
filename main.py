#!/usr/bin/env python3
"""main.py - Main entry point and CLI interface for the Human Fall & Lying Detection System.

Usage Examples:
  # 1. Run live detection on a video and save annotated output:
  python main.py run --source data/videos/urfd/fall-01-cam0.mp4 --out output.mp4

  # 2. Run benchmark evaluation on a manifest:
  python main.py benchmark --manifest data/manifest_urfd.csv --output-dir benchmark_results_urfd

  # 3. Generate published demo videos:
  python main.py demo

  # 4. Check system hardware & model status:
  python main.py info
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import (
    FallMonitor,
    load_cfg,
    compute_features,
    evaluate_video_manifest,
    TSSTGClassifier,
    CLASS_NAMES,
)

# Visual styling constants
COLOR_NORMAL = (40, 200, 40)       # Green
COLOR_FALL = (30, 30, 235)         # Red
COLOR_LYING = (0, 150, 255)        # Amber / Orange
COLOR_BG_DARK = (20, 20, 20)

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6),                                    # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # Arms
    (5, 11), (6, 12), (11, 12),                # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
]


def draw_hud(
    canvas: np.ndarray,
    video_title: str,
    t: float,
    status: str,
    action_probs: Dict[str, float],
    kinematics: Dict[str, float],
    status_color: Tuple[int, int, int]
):
    """Draws a real-time telemetry HUD overlay at the top of the frame."""
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    # Upper telemetry banner
    cv2.rectangle(overlay, (0, 0), (w, 64), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    # Title & time
    cv2.putText(canvas, f"AI FALL & LYING MONITOR | {video_title}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"T: {t:05.2f}s", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 230, 255), 2, cv2.LINE_AA)

    # Status Pill / Badge
    pill_w, pill_h = 240, 34
    pill_x = max(10, w - pill_w - 12)
    pill_y = 14
    cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), status_color, -1)
    cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (255, 255, 255), 1)

    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    tx = pill_x + (pill_w - text_size[0]) // 2
    ty = pill_y + (pill_h + text_size[1]) // 2 - 2
    cv2.putText(canvas, status, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Telemetry metrics row
    p_fall = action_probs.get("Fall Down", 0.0)
    p_walk = action_probs.get("Walking", 0.0)
    angle = kinematics.get("angle", 0.0)
    aspect = kinematics.get("aspect", 0.0)

    telem_str = f"Fall Prob: {p_fall:4.1%} | Walk: {p_walk:4.1%} | Angle: {angle:4.1f}° | Aspect: {aspect:3.2f}"
    cv2.putText(canvas, telem_str, (120, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 220), 1, cv2.LINE_AA)


def draw_person(
    canvas: np.ndarray,
    box: np.ndarray,
    kps: np.ndarray,
    tid: int,
    state: str,
    state_color: Tuple[int, int, int]
):
    """Draws keypoints, skeleton links, bounding box, and floating tag on top of person."""
    bx1, by1, bx2, by2 = box.astype(int)

    # Draw skeleton bones
    for p1_idx, p2_idx in SKELETON_PAIRS:
        c1, c2 = kps[p1_idx, 2], kps[p2_idx, 2]
        if c1 >= 0.25 and c2 >= 0.25:
            pt1 = (int(kps[p1_idx, 0]), int(kps[p1_idx, 1]))
            pt2 = (int(kps[p2_idx, 0]), int(kps[p2_idx, 1]))
            cv2.line(canvas, pt1, pt2, (120, 220, 255), 2, cv2.LINE_AA)

    # Draw keypoint dots
    for j in range(17):
        if kps[j, 2] >= 0.25:
            pt = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.circle(canvas, pt, 4, state_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw person bounding box
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), state_color, 2, cv2.LINE_AA)

    # Floating Tag directly on top of person
    tag_h = 24
    tag_w = max(180, (bx2 - bx1))
    tag_x1 = max(4, bx1)
    tag_y1 = max(68, by1 - tag_h - 4)
    tag_x2 = tag_x1 + tag_w
    tag_y2 = tag_y1 + tag_h

    overlay = canvas.copy()
    cv2.rectangle(overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), state_color, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.rectangle(canvas, (tag_x1, tag_y1), (tag_x2, tag_y2), (255, 255, 255), 1)

    label_text = f"ID #{tid} | {state}"
    cv2.putText(canvas, label_text, (tag_x1 + 6, tag_y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)


def run_pipeline(
    source: str,
    output_path: Optional[str] = None,
    model_path: str = "weights/yolo26x-pose.pt",
    tracker_config: str = "bytetrack.yaml",
    device: str = "cuda:0",
    conf: float = 0.25,
    show: bool = False
):
    """Executes the complete pose tracking and fall/lying detection pipeline on video or camera."""
    from ultralytics import YOLO

    device_str = device if (torch.cuda.is_available() and "cuda" in device) else "cpu"
    print(f"[*] Initializing YOLO pose model from {model_path} on {device_str}...")
    yolo_model = YOLO(model_path)

    # Open video capture
    is_cam = source.isdigit()
    cap_src = int(source) if is_cam else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[!] Error: Could not open source '{source}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    print(f"[*] Input Source: {source} ({w}x{h} @ {fps:.1f} fps, {total_frames} frames)")

    cfg = load_cfg()
    monitor = FallMonitor(cfg=cfg, img_shape=(h, w), device=device_str)

    writer = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        print(f"[*] Recording output to: {output_path}")

    active_fall_alert = False
    active_lying_alert = False
    alert_hold_frames = 0
    idx = 0

    print("[*] Processing started... Press 'q' to stop.")
    t0 = time.time()

    try:
        for res in yolo_model.track(
            source=cap_src,
            tracker=tracker_config,
            conf=conf,
            stream=True,
            verbose=False,
            device=device_str
        ):
            t = idx / fps
            idx += 1
            frame = res.orig_img.copy()

            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
            ids = res.boxes.id.cpu().numpy().astype(int) if (res.boxes is not None and res.boxes.id is not None) else np.array([1] * len(boxes))
            kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), 17, 3), dtype=np.float32)

            events = monitor.update(t, ids, boxes, kps, frame=res.orig_img)

            # Handle events
            for ev in events:
                ev_type = ev.get("type")
                t_cand = ev.get("t_candidate", t)
                if ev_type == "fall":
                    active_fall_alert = True
                    active_lying_alert = False
                    alert_hold_frames = int(fps * 4.0)
                    print(f"  [T={t:5.2f}s] >>> [ALERT] FALL DETECTED on Track {ev.get('track')} (cand={t_cand:.2f}s)")
                elif ev_type == "lying":
                    active_lying_alert = True
                    alert_hold_frames = int(fps * 4.0)
                    print(f"  [T={t:5.2f}s] >>> [ALERT] LYING DETECTED on Track {ev.get('track')} (cand={t_cand:.2f}s)")

            if alert_hold_frames > 0:
                alert_hold_frames -= 1
                if alert_hold_frames == 0 and not active_lying_alert:
                    active_fall_alert = False

            action_probs: Dict[str, float] = {}
            kinematics: Dict[str, float] = {}

            if len(ids) > 0:
                tid = int(ids[0])
                b = boxes[0]
                k = kps[0]

                tr_data = monitor.tracks.get(tid)
                if tr_data and len(tr_data["t"]) >= 3:
                    try:
                        feat = compute_features(
                            dict(id=tid, t=np.array(tr_data["t"]), kp=np.array(tr_data["kp"]), box=np.array(tr_data["box"])),
                            cfg,
                            ref_scale=tr_data["ref_scale"]
                        )
                        kinematics["angle"] = float(feat["angle"][-1]) if np.isfinite(feat["angle"][-1]) else 0.0
                        kinematics["aspect"] = float(feat["aspect"][-1]) if np.isfinite(feat["aspect"][-1]) else 0.0
                    except Exception:
                        pass

                if tr_data and tr_data.get("probs_hist"):
                    latest_p = tr_data["probs_hist"][-1][1]
                    action_probs["Walking"] = float(latest_p[1]) if len(latest_p) > 1 else 0.0
                    action_probs["Fall Down"] = float(latest_p[6]) if len(latest_p) > 6 else 0.0

                if active_fall_alert:
                    person_state = "ALERT: FALL DETECTED"
                    person_color = COLOR_FALL
                elif active_lying_alert:
                    person_state = "ALERT: LYING DETECTED"
                    person_color = COLOR_LYING
                else:
                    person_state = "NORMAL (Upright)"
                    person_color = COLOR_NORMAL

                draw_person(frame, b, k, tid, person_state, person_color)

            if active_fall_alert:
                hud_status = "CRITICAL: FALL DETECTED"
                hud_color = COLOR_FALL
            elif active_lying_alert:
                hud_status = "MONITOR: LYING ON FLOOR"
                hud_color = COLOR_LYING
            else:
                hud_status = "NORMAL: MONITORING"
                hud_color = COLOR_NORMAL

            display_title = Path(source).name if not is_cam else f"Camera #{source}"
            draw_hud(frame, display_title, t, hud_status, action_probs, kinematics, hud_color)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("Human Fall & Lying Detection Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"[*] Processed {idx} frames in {elapsed:.2f}s ({idx/elapsed:.1f} fps).")


def run_benchmark(args):
    """Invokes benchmark evaluation."""
    print(f"[*] Starting benchmark on manifest: {args.manifest}")
    metrics = evaluate_video_manifest(
        manifest_path=args.manifest,
        model_name=args.model,
        tracker=args.tracker,
        tolerance_s=args.tolerance,
        output_dir=args.output_dir,
        conf_det=args.conf_det,
        imgsz=args.imgsz,
        stride=args.stride,
        device=args.device,
    )
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Videos Evaluated:        {metrics['num_videos']}")
    print(f"  Duration:                {metrics['total_duration_hours']:.2f} hours")
    print(f"  Fall Precision / Recall: {metrics['precision']:.4f} / {metrics['recall']:.4f} (F1: {metrics['f1']:.4f})")
    print(f"  Fall Detection Latency:  {metrics['latency_s']:.2f} s")
    lying_m = metrics.get("lying_metrics", {})
    if lying_m:
        print(f"  Lying Precision/Recall:  {lying_m['precision']:.4f} / {lying_m['recall']:.4f} (F1: {lying_m['f1']:.4f})")
        print(f"  Lying Detection Latency: {lying_m['latency_s']:.2f} s")
    print("=" * 60)


def run_demos():
    """Generates the demonstration videos."""
    from scripts import create_demos
    create_demos.main()


def print_system_info():
    """Prints environment, PyTorch, GPU and weights status."""
    print("=" * 60)
    print("SYSTEM DIAGNOSTICS & HARDWARE STATUS")
    print("=" * 60)
    print(f"Python Executable:    {sys.executable}")
    print(f"PyTorch Version:      {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:       {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device:           {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM:           {mem_gb:.2f} GB")

    print("\nModel Weights in weights/:")
    w_dir = ROOT / "weights"
    for w in sorted(w_dir.glob("*.pt*")):
        sz_mb = w.stat().st_size / (1024 * 1024)
        print(f"  - {w.name} ({sz_mb:.1f} MB)")

    print("\nDatasets in data/:")
    for dname in ["videos/le2i", "videos/urfd", "demos"]:
        p = ROOT / "data" / dname
        if p.exists():
            cnt = len(list(p.glob("*.*")))
            print(f"  - data/{dname}: {cnt} files")
    print("=" * 60)


def build_cli():
    parser = argparse.ArgumentParser(
        description="Human Fall & Persistent Lying Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: run
    p_run = sub.add_parser("run", help="Run detection on a video file or camera stream")
    p_run.add_argument("--source", type=str, default="data/videos/urfd/fall-01-cam0.mp4", help="Video path or camera index (default: '0')")
    p_run.add_argument("--out", type=str, default=None, help="Path to save annotated MP4 video")
    p_run.add_argument("--model", type=str, default="weights/yolo26x-pose.pt", help="YOLO pose model path")
    p_run.add_argument("--tracker", type=str, default="bytetrack.yaml", help="ByteTrack YAML config")
    p_run.add_argument("--device", type=str, default="cuda:0", help="Inference device ('cuda:0' or 'cpu')")
    p_run.add_argument("--conf", type=float, default=0.25, help="Pose detection confidence threshold")
    p_run.add_argument("--show", action="store_true", help="Display live OpenCV window")

    # Command: benchmark
    p_bm = sub.add_parser("benchmark", help="Run evaluation benchmark on a dataset manifest")
    p_bm.add_argument("--manifest", type=str, default="data/manifest_urfd.csv", help="Path to manifest CSV/JSON")
    p_bm.add_argument("--model", type=str, default="weights/yolo26x-pose.pt", help="YOLO pose model path")
    p_bm.add_argument("--tracker", type=str, default="bytetrack.yaml", help="ByteTrack YAML config")
    p_bm.add_argument("--output-dir", type=str, default="benchmark_results_urfd", help="Directory for output report")
    p_bm.add_argument("--tolerance", type=float, default=2.0, help="GT match tolerance in seconds")
    p_bm.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    p_bm.add_argument("--stride", type=int, default=1, help="Frame stride")
    p_bm.add_argument("--conf-det", type=float, default=0.25, help="Detector confidence")
    p_bm.add_argument("--imgsz", type=int, default=640, help="Inference resolution")

    # Command: demo
    sub.add_parser("demo", help="Generate the fall and lying demo videos")

    # Command: info
    sub.add_parser("info", help="Print system diagnostics and model status")

    return parser


def main():
    parser = build_cli()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command == "run":
        run_pipeline(
            source=args.source,
            output_path=args.out,
            model_path=args.model,
            tracker_config=args.tracker,
            device=args.device,
            conf=args.conf,
            show=args.show
        )
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "demo":
        run_demos()
    elif args.command == "info":
        print_system_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
