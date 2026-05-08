from __future__ import annotations

import shutil
from pathlib import Path

from src.auto_label.label_hierarchy import fine_to_coarse, make_display_label
from scripts.auto_label.evaluate_yolo_hierarchy import evaluate_label_dirs


def test_label_hierarchy_display_names():
    expected = {
        "pot": "cookware:pot",
        "pan": "cookware:pan",
        "lid": "cookware:lid",
        "carton": "container:carton",
        "olive": "ingredient:olive",
        "sponge": "kitchen_scene:sponge",
        "unknown_object": "unknown:unknown_object",
    }
    for label, display in expected.items():
        assert make_display_label(label) == display


def test_coarse_groups():
    assert fine_to_coarse("pot") == "cookware"
    assert fine_to_coarse("carton") == "container"
    assert fine_to_coarse("sponge") == "kitchen_scene"


def test_metric_fine_and_coarse_accuracy():
    root = Path("tests") / "_tmp_label_hierarchy"
    if root.exists():
        shutil.rmtree(root)
    gt = root / "gt"
    pred = root / "pred"
    gt.mkdir(parents=True)
    pred.mkdir(parents=True)
    names = ["pot", "pan", "carton", "box", "sponge"]
    (gt / "a.txt").write_text("0 0 0 1 0 1 1\n2 0 0 1 0 1 1\n4 0 0 1 0 1 1\n", encoding="utf-8")
    (pred / "a.txt").write_text("1 0 0 1 0 1 1\n3 0 0 1 0 1 1\n4 0 0 1 0 1 1\n", encoding="utf-8")

    report = evaluate_label_dirs(gt, pred, names)

    assert report["overall"]["fine_correct"] == 1
    assert report["overall"]["coarse_correct"] == 3
    assert report["overall"]["fine_accuracy"] == 1 / 3
    assert report["overall"]["coarse_accuracy"] == 1.0
    shutil.rmtree(root)
