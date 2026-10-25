"""T16: evaluate_benchmark.py - Le2i annotation parser + held-out metrics (plan Task 5)."""
import numpy as np

import evaluate_benchmark as EB  # ModuleNotFoundError == expected red


def test_le2i_annotation_parser_and_heldout_benchmark_metrics(tmp_path):
    ann = tmp_path / "video (1).txt"
    ann.write_text("10\n20\n10,1,100,100,200,300\n11,1,101,101,201,301\n")
    r = EB.parse_le2i_annotation(ann, fps=25.0)
    assert r["has_fall"] is True
    assert (r["fall_start_frame"], r["fall_end_frame"]) == (10, 20)
    assert np.isclose(r["fall_start_s"], 0.4) and np.isclose(r["fall_end_s"], 0.8)

    none = tmp_path / "video (2).txt"
    none.write_text("0\n0\n1,1,5,5,10,10\n")
    r0 = EB.parse_le2i_annotation(none, fps=25.0)
    assert r0["has_fall"] is False and r0["fall_start_s"] is None

    recs = [
        dict(video="f1", gt_fall=True, activity=None, detected=True, latency_s=0.5, false_alarms=0),
        dict(video="f2", gt_fall=True, activity=None, detected=True, latency_s=1.5, false_alarms=1),
        dict(video="f3", gt_fall=True, activity=None, detected=False, latency_s=None, false_alarms=0),
        dict(video="b1", gt_fall=False, activity="bending", detected=False, latency_s=None, false_alarms=2),
        dict(video="k1", gt_fall=False, activity="kneeling", detected=False, latency_s=None, false_alarms=0),
        dict(video="s1", gt_fall=False, activity="sitting", detected=False, latency_s=None, false_alarms=1),
        dict(video="w1", gt_fall=False, activity="floor_work", detected=False, latency_s=None, false_alarms=0),
    ]
    m = EB.compute_benchmark_metrics(recs, total_duration_hours=0.5)
    assert np.isclose(m["precision"], 2 / 6) and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["f1"], 4 / 9)
    assert np.isclose(m["fa_per_hour"], 8.0)
    assert np.isclose(m["latency_s"], 1.0)
    assert m["lookalike_fa_breakdown"] == {"bending": 2, "kneeling": 0, "sitting": 1, "floor_work": 0}
    assert callable(EB.evaluate_video_manifest)
