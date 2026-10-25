"""omnifall_adapter.py - Ingests OmniFall annotations and builds standardized manifests.

Maps human-verified temporal annotations from the OmniFall benchmark
(simplexsigil2/omnifall) onto local datasets (Le2i, OF-Syn, URFD, etc.).
Generates per-video ground-truth JSON files and standardized manifest CSVs
compatible with evaluate_benchmark.py.
"""
import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("omnifall_adapter")

# OmniFall 16-class action taxonomy (LABELS.md / label2id.csv)
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "walk",
    1: "fall",
    2: "fallen",
    3: "sit_down",
    4: "sitting",
    5: "lie_down",
    6: "lying",
    7: "stand_up",
    8: "standing",
    9: "other",
    10: "kneel_down",
    11: "kneeling",
    12: "squat_down",
    13: "squatting",
    14: "crawl",
    15: "jump",
}

CLASS_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items()}


def is_fall_class(label_id: int) -> bool:
    """Class 1: Dynamic act of falling."""
    return label_id == 1


def is_lying_class(label_id: int) -> bool:
    """Class 2 (fallen - post-fall) and Class 6 (lying - intentional floor/bed lying)."""
    return label_id in (2, 6)


def is_sit_lookalike(label_id: int) -> bool:
    """Class 3 (sit_down) and Class 4 (sitting)."""
    return label_id in (3, 4)


def is_floor_work(label_id: int) -> bool:
    """Classes 10..14 (kneel, squat, crawl)."""
    return label_id in (10, 11, 12, 13, 14)


def merge_intervals(intervals: List[Tuple[float, float]], gap_tolerance_s: float = 0.5) -> List[List[float]]:
    """Merges contiguous or overlapping time intervals [start, end]."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [[float(sorted_ivs[0][0]), float(sorted_ivs[0][1])]]
    for cur_s, cur_e in sorted_ivs[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e + gap_tolerance_s:
            merged[-1][1] = max(prev_e, float(cur_e))
        else:
            merged.append([float(cur_s), float(cur_e)])
    return merged


def resolve_le2i_video_path(rel_path: str, video_root: Path) -> Optional[Path]:
    """Resolves an OmniFall Le2i path (e.g. 'Coffee_room_01/video_14') to disk.

    Handles variations in directory nesting and filename patterns:
      - 'data/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (14).avi'
      - 'data/videos/le2i/Lecture_room/Lecture room/video (14).avi'
    """
    video_root = Path(video_root)
    if not video_root.exists():
        return None

    # Direct match if standard layout
    direct_candidates = [
        video_root / f"{rel_path}.avi",
        video_root / f"{rel_path}.mp4",
        video_root / f"{rel_path}.mkv",
    ]
    for cand in direct_candidates:
        if cand.exists():
            return cand

    parts = rel_path.replace("\\", "/").strip("/").split("/")
    if len(parts) < 2:
        return None

    room_key = parts[0].lower().replace("_", "").replace(" ", "")
    vid_part = parts[1]
    m = re.search(r"video[_\s]*(\d+)", vid_part, re.IGNORECASE)
    if not m:
        return None
    target_num = int(m.group(1))

    # Search video_root recursively
    for f in video_root.rglob("*.avi"):
        parent_str = str(f.parent.parent).lower().replace("_", "").replace(" ", "")
        if room_key in parent_str or room_key in str(f.parent).lower().replace("_", "").replace(" ", ""):
            match_f = re.search(r"video\s*\((\d+)\)", f.name, re.IGNORECASE)
            if match_f and int(match_f.group(1)) == target_num:
                return f
            match_f2 = re.search(r"video_(\d+)", f.name, re.IGNORECASE)
            if match_f2 and int(match_f2.group(1)) == target_num:
                return f

    return None


_GENERIC_CACHE: Dict[Path, Dict[str, Path]] = {}


def resolve_generic_video_path(rel_path: str, video_root: Path) -> Optional[Path]:
    """Resolves arbitrary dataset video paths (OF-Syn, URFD, etc.)."""
    video_root = Path(video_root)
    for ext in ("", ".mp4", ".avi", ".mkv", ".webm"):
        cand = video_root / f"{rel_path}{ext}"
        if cand.exists() and cand.is_file():
            return cand

    # Check under video/ subfolder if following OmniFall layout {dataset}/video/{path}.mp4
    for ext in (".mp4", ".avi"):
        cand = video_root / "video" / f"{rel_path}{ext}"
        if cand.exists() and cand.is_file():
            return cand

    # Fast in-memory dictionary lookup
    if video_root not in _GENERIC_CACHE:
        _GENERIC_CACHE[video_root] = {
            f.stem.lower(): f for f in video_root.rglob("*") if f.is_file()
        }
    return _GENERIC_CACHE[video_root].get(Path(rel_path).name.lower())


def parse_omnifall_label_csv(label_csv_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Parses OmniFall label CSV and groups temporal segments by video path."""
    csv_path = Path(label_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OmniFall label CSV not found: {csv_path}")

    segments_by_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or "path" not in row or "label" not in row:
                continue
            path_key = row["path"].strip()
            try:
                lbl = int(row["label"])
                s = float(row["start"])
                e = float(row["end"])
            except (ValueError, TypeError):
                continue

            entry: Dict[str, Any] = {
                "path": path_key,
                "label": lbl,
                "label_name": CLASS_ID_TO_NAME.get(lbl, "unknown"),
                "start": s,
                "end": e,
                "subject": int(row.get("subject", -1)) if row.get("subject") else -1,
                "cam": int(row.get("cam", -1)) if row.get("cam") else -1,
                "dataset": row.get("dataset", ""),
            }
            # Demographic / scene metadata for of-syn if present
            for extra in (
                "age_group", "gender_presentation", "monk_skin_tone", "race_ethnicity_omb",
                "bmi_band", "height_band", "environment_category", "camera_shot",
                "speed", "camera_elevation", "camera_azimuth", "camera_distance"
            ):
                if extra in row:
                    entry[extra] = row[extra]

            segments_by_path[path_key].append(entry)

    return dict(segments_by_path)


def build_video_annotation(
    path_key: str,
    segments: List[Dict[str, Any]],
    video_resolved_path: Optional[Path] = None,
    fps: float = 25.0
) -> Dict[str, Any]:
    """Transforms raw OmniFall segments for a single video into standard ground truth."""
    fall_segments = [s for s in segments if is_fall_class(s["label"])]
    lying_segments = [s for s in segments if is_lying_class(s["label"])]

    has_fall = len(fall_segments) > 0
    fall_start_s: Optional[float] = None
    fall_end_s: Optional[float] = None
    fall_start_frame: int = 0
    fall_end_frame: int = 0

    if has_fall:
        # Sort by start time, pick primary fall
        primary_fall = sorted(fall_segments, key=lambda x: x["start"])[0]
        fall_start_s = float(primary_fall["start"])
        fall_end_s = float(primary_fall["end"])
        fall_start_frame = int(round(fall_start_s * fps))
        fall_end_frame = int(round(fall_end_s * fps))

    # Lying intervals
    lying_raw_intervals = [(s["start"], s["end"]) for s in lying_segments]
    lying_intervals_s = merge_intervals(lying_raw_intervals)
    has_lying = len(lying_intervals_s) > 0

    # Look-alikes summary
    has_sitting = any(is_sit_lookalike(s["label"]) for s in segments)
    has_floor = any(is_floor_work(s["label"]) for s in segments)
    has_standing = any(s["label"] == 8 for s in segments)
    has_walking = any(s["label"] == 0 for s in segments)

    labels_present = sorted(list({s["label_name"] for s in segments}))

    return {
        "video_key": path_key,
        "video_file": str(video_resolved_path) if video_resolved_path else None,
        "has_fall": has_fall,
        "fall_start_s": fall_start_s,
        "fall_end_s": fall_end_s,
        "fall_start_frame": fall_start_frame,
        "fall_end_frame": fall_end_frame,
        "has_lying": has_lying,
        "lying_intervals_s": lying_intervals_s,
        "labels_present": labels_present,
        "lookalikes": {
            "sitting": has_sitting,
            "floor_work": has_floor,
            "standing": has_standing,
            "walking": has_walking,
        },
        "segments": segments,
    }


def generate_omnifall_manifest(
    label_csv_path: Union[str, Path],
    video_root: Union[str, Path],
    output_ann_dir: Union[str, Path],
    output_manifest_path: Union[str, Path],
    dataset_name: str = "le2i",
    project_root: Optional[Union[str, Path]] = None,
    fps: float = 25.0
) -> Tuple[int, int, Path]:
    """Generates per-video ground-truth JSON files and standardized manifest CSV.

    Args:
        label_csv_path: Path to OmniFall label CSV (e.g. data/omnifall/labels/le2i.csv).
        video_root: Directory where dataset videos reside on disk.
        output_ann_dir: Directory where JSON annotations will be saved.
        output_manifest_path: Destination manifest CSV file.
        dataset_name: Dataset identifier ('le2i', 'of-syn', etc.).
        project_root: Root directory for relative path serialization.
        fps: Video FPS for frame calculation.

    Returns:
        Tuple of (num_videos, num_falls, manifest_path).
    """
    label_csv_path = Path(label_csv_path)
    video_root = Path(video_root)
    output_ann_dir = Path(output_ann_dir)
    output_manifest_path = Path(output_manifest_path)
    proj_root = Path(project_root) if project_root else output_manifest_path.parent.parent

    output_ann_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = parse_omnifall_label_csv(label_csv_path)
    logger.info("Loaded %d unique videos from %s", len(grouped), label_csv_path.name)

    manifest_rows: List[Dict[str, str]] = []
    unresolved_count = 0
    total_falls = 0

    for path_key, segs in sorted(grouped.items()):
        if dataset_name.lower() == "le2i":
            vpath = resolve_le2i_video_path(path_key, video_root)
        else:
            vpath = resolve_generic_video_path(path_key, video_root)

        if vpath is None or not vpath.exists():
            logger.warning("Could not resolve video file for key: %s (in %s)", path_key, video_root)
            unresolved_count += 1
            continue

        ann = build_video_annotation(path_key, segs, video_resolved_path=vpath, fps=fps)
        if ann["has_fall"]:
            total_falls += 1

        # Save JSON annotation
        safe_key = path_key.replace("/", "_").replace("\\", "_")
        ann_filename = f"{safe_key}.json"
        ann_filepath = output_ann_dir / ann_filename
        ann_filepath.write_text(json.dumps(ann, indent=2), encoding="utf-8")

        # Determine activity tag
        activity = "fall" if ann["has_fall"] else "normal"
        if not ann["has_fall"]:
            if ann["lookalikes"]["sitting"]:
                activity = "sitting"
            elif ann["lookalikes"]["floor_work"]:
                activity = "floor_work"
            elif ann["lookalikes"]["walking"]:
                activity = "walking"

        # Serialize relative paths relative to proj_root or output_manifest_path.parent
        try:
            rel_vpath = vpath.relative_to(proj_root / "data")
        except ValueError:
            try:
                rel_vpath = vpath.relative_to(proj_root)
            except ValueError:
                rel_vpath = vpath

        try:
            rel_ann = ann_filepath.relative_to(proj_root / "data")
        except ValueError:
            try:
                rel_ann = ann_filepath.relative_to(proj_root)
            except ValueError:
                rel_ann = ann_filepath

        manifest_rows.append({
            "video_path": str(rel_vpath).replace("\\", "/"),
            "annotation_path": str(rel_ann).replace("\\", "/"),
            "activity": activity,
        })

    # Write CSV manifest
    with output_manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "annotation_path", "activity"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        "Generated manifest with %d videos (%d falls, %d normal). Unresolved: %d. Saved -> %s",
        len(manifest_rows),
        total_falls,
        len(manifest_rows) - total_falls,
        unresolved_count,
        output_manifest_path
    )
    return len(manifest_rows), total_falls, output_manifest_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="OmniFall Annotation Adapter and Manifest Generator")
    parser.add_argument("--labels", type=str, default="data/omnifall/labels/le2i.csv", help="Path to OmniFall label CSV")
    parser.add_argument("--videos", type=str, default="data/videos/le2i", help="Directory with video files")
    parser.add_argument("--dataset", type=str, default="le2i", choices=["le2i", "of-syn", "generic"], help="Dataset type")
    parser.add_argument("--output-dir", type=str, default="data/omnifall/annotations/le2i", help="Output annotations directory")
    parser.add_argument("--manifest", type=str, default="data/manifest_omnifall_le2i.csv", help="Output manifest CSV path")
    args = parser.parse_args()

    generate_omnifall_manifest(
        label_csv_path=Path(args.labels),
        video_root=Path(args.videos),
        output_ann_dir=Path(args.output_dir),
        output_manifest_path=Path(args.manifest),
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
