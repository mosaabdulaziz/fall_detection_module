"""test_omnifall_adapter.py - Unit tests for OmniFall annotation parsing and manifest creation."""
import json
from pathlib import Path

import pytest
from src.omnifall_adapter import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    build_video_annotation,
    generate_omnifall_manifest,
    is_fall_class,
    is_floor_work,
    is_lying_class,
    is_sit_lookalike,
    merge_intervals,
    parse_omnifall_label_csv,
    resolve_generic_video_path,
    resolve_le2i_video_path,
)


def test_taxonomy_definitions():
    """Verify class taxonomy conforms to OmniFall LABELS.md standard."""
    assert CLASS_ID_TO_NAME[0] == "walk"
    assert CLASS_ID_TO_NAME[1] == "fall"
    assert CLASS_ID_TO_NAME[2] == "fallen"
    assert CLASS_ID_TO_NAME[3] == "sit_down"
    assert CLASS_ID_TO_NAME[4] == "sitting"
    assert CLASS_ID_TO_NAME[5] == "lie_down"
    assert CLASS_ID_TO_NAME[6] == "lying"
    assert CLASS_ID_TO_NAME[7] == "stand_up"
    assert CLASS_ID_TO_NAME[8] == "standing"
    assert CLASS_ID_TO_NAME[9] == "other"
    assert CLASS_ID_TO_NAME[10] == "kneel_down"
    assert CLASS_ID_TO_NAME[15] == "jump"

    assert is_fall_class(1) is True
    assert is_fall_class(0) is False
    assert is_fall_class(2) is False

    assert is_lying_class(2) is True
    assert is_lying_class(6) is True
    assert is_lying_class(1) is False

    assert is_sit_lookalike(3) is True
    assert is_sit_lookalike(4) is True
    assert is_sit_lookalike(0) is False

    assert is_floor_work(10) is True
    assert is_floor_work(12) is True
    assert is_floor_work(14) is True
    assert is_floor_work(8) is False


def test_interval_merging():
    """Verify temporal interval merging with gap tolerance."""
    # Overlapping
    ivs = [(1.0, 3.0), (2.5, 5.0)]
    merged = merge_intervals(ivs)
    assert merged == [[1.0, 5.0]]

    # Close gap (within 0.5s tolerance)
    ivs2 = [(1.0, 3.0), (3.3, 5.0)]
    merged2 = merge_intervals(ivs2, gap_tolerance_s=0.5)
    assert merged2 == [[1.0, 5.0]]

    # Distant gap (> 0.5s)
    ivs3 = [(1.0, 3.0), (4.0, 6.0)]
    merged3 = merge_intervals(ivs3, gap_tolerance_s=0.5)
    assert merged3 == [[1.0, 3.0], [4.0, 6.0]]

    # Empty
    assert merge_intervals([]) == []


def test_build_video_annotation_fall():
    """Verify single video annotation with fall and post-fall lying."""
    segments = [
        {"path": "vid1", "label": 0, "label_name": "walk", "start": 0.0, "end": 2.0},
        {"path": "vid1", "label": 1, "label_name": "fall", "start": 2.1, "end": 3.5},
        {"path": "vid1", "label": 2, "label_name": "fallen", "start": 3.5, "end": 8.0},
        {"path": "vid1", "label": 7, "label_name": "stand_up", "start": 8.0, "end": 10.0},
    ]

    ann = build_video_annotation("vid1", segments, fps=25.0)
    assert ann["has_fall"] is True
    assert ann["fall_start_s"] == 2.1
    assert ann["fall_end_s"] == 3.5
    assert ann["fall_start_frame"] == 52  # round(2.1 * 25)
    assert ann["fall_end_frame"] == 88  # round(3.5 * 25)
    assert ann["has_lying"] is True
    assert ann["lying_intervals_s"] == [[3.5, 8.0]]
    assert "fall" in ann["labels_present"]
    assert "fallen" in ann["labels_present"]


def test_build_video_annotation_normal_lookalike():
    """Verify normal ADL video with sitting and kneeling look-alikes."""
    segments = [
        {"path": "vid2", "label": 0, "label_name": "walk", "start": 0.0, "end": 3.0},
        {"path": "vid2", "label": 3, "label_name": "sit_down", "start": 3.0, "end": 4.5},
        {"path": "vid2", "label": 4, "label_name": "sitting", "start": 4.5, "end": 12.0},
    ]

    ann = build_video_annotation("vid2", segments, fps=25.0)
    assert ann["has_fall"] is False
    assert ann["fall_start_s"] is None
    assert ann["fall_end_s"] is None
    assert ann["has_lying"] is False
    assert ann["lookalikes"]["sitting"] is True
    assert ann["lookalikes"]["walking"] is True


def test_le2i_path_resolution(tmp_path):
    """Test resolution of complex nested Le2i directory naming."""
    # Create fake directory layout
    room = tmp_path / "Coffee_room_01" / "Coffee_room_01" / "Videos"
    room.mkdir(parents=True, exist_ok=True)
    vid_file = room / "video (14).avi"
    vid_file.write_bytes(b"dummy video data")

    resolved = resolve_le2i_video_path("Coffee_room_01/video_14", tmp_path)
    assert resolved is not None
    assert resolved.name == "video (14).avi"

    # Test non-existent video
    resolved_none = resolve_le2i_video_path("Coffee_room_01/video_999", tmp_path)
    assert resolved_none is None


def test_manifest_generation(tmp_path):
    """Test generating manifest and json files from a mock CSV."""
    label_csv = tmp_path / "mock_labels.csv"
    label_csv.write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "Room1/video_1,0,0.0,2.0,1,1,le2i\n"
        "Room1/video_1,1,2.0,3.5,1,1,le2i\n"
        "Room1/video_2,0,0.0,5.0,2,1,le2i\n",
        encoding="utf-8"
    )

    vid_dir = tmp_path / "videos" / "Room1"
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "video (1).avi").write_bytes(b"vid1")
    (vid_dir / "video (2).avi").write_bytes(b"vid2")

    ann_dir = tmp_path / "annotations"
    manifest_csv = tmp_path / "data" / "manifest_test.csv"

    n_vids, n_falls, m_path = generate_omnifall_manifest(
        label_csv_path=label_csv,
        video_root=tmp_path / "videos",
        output_ann_dir=ann_dir,
        output_manifest_path=manifest_csv,
        dataset_name="le2i",
        project_root=tmp_path,
    )

    assert n_vids == 2
    assert n_falls == 1
    assert m_path.exists()

    # Check generated JSON annotation content
    ann1 = json.loads((ann_dir / "Room1_video_1.json").read_text(encoding="utf-8"))
    assert ann1["has_fall"] is True
    assert ann1["fall_start_s"] == 2.0
    assert ann1["fall_end_s"] == 3.5

    ann2 = json.loads((ann_dir / "Room1_video_2.json").read_text(encoding="utf-8"))
    assert ann2["has_fall"] is False


def test_real_omnifall_le2i_dataset_resolution():
    """Verify that all 190 Le2i videos on local disk map to OmniFall labels with zero missing."""
    labels_csv = Path("data/omnifall/labels/le2i.csv")
    video_root = Path("data/videos/le2i")

    if not (labels_csv.exists() and video_root.exists()):
        pytest.skip("Local Le2i video or OmniFall labels not found on disk.")

    grouped = parse_omnifall_label_csv(labels_csv)
    assert len(grouped) == 190

    unresolved = []
    for path_key in grouped:
        resolved = resolve_le2i_video_path(path_key, video_root)
        if resolved is None or not resolved.exists():
            unresolved.append(path_key)

    assert len(unresolved) == 0, f"Unresolved Le2i videos: {unresolved}"
