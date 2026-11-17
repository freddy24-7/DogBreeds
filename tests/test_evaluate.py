"""Stage 4: the metrics, tested on tiny made-up examples where the answer is known."""
import json
from pathlib import Path

import numpy as np
import pytest

from dogbreeds.data import STANFORD, stanford_breeds
from dogbreeds.evaluate import calibration, confused_pairs, confusion_matrix, per_breed_accuracy, topk_accuracy
from dogbreeds.predictors import EFFNET_B0

# 4 photos, 3 breeds. Rows: the model's probabilities. Labels: the truth.
PROBS = np.array([
    [0.7, 0.2, 0.1],   # true 0, predicted 0: right
    [0.6, 0.3, 0.1],   # true 1, predicted 0: wrong, but 1 is 2nd
    [0.1, 0.1, 0.8],   # true 2, predicted 2: right
    [0.5, 0.4, 0.1],   # true 1, predicted 0: wrong again
])
LABELS = np.array([0, 1, 2, 1])


def test_topk_accuracy():
    assert topk_accuracy(PROBS, LABELS, 1) == 0.5
    assert topk_accuracy(PROBS, LABELS, 2) == 1.0


def test_per_breed_accuracy():
    assert per_breed_accuracy(PROBS, LABELS, 3).tolist() == [1.0, 0.0, 1.0]


def test_confusion_and_pairs():
    cm = confusion_matrix(PROBS, LABELS, 3)
    assert cm[1, 0] == 2 and cm.trace() == 2 and cm.sum() == 4
    assert confused_pairs(cm) == [(1, 0, 2)]   # breed 1 mistaken for breed 0, twice


def test_calibration_of_a_perfectly_calibrated_model_is_zero():
    # 10 photos, all 70% sure, 7 of them right -> confidence 0.7 = accuracy 0.7
    probs = np.tile([0.7, 0.3], (10, 1))
    labels = np.array([0] * 7 + [1] * 3)
    _, ece = calibration(probs, labels)
    assert ece == pytest.approx(0.0)


def test_overconfident_model_has_positive_ece():
    probs = np.tile([0.9, 0.1], (10, 1))
    labels = np.array([0] * 5 + [1] * 5)   # 90% sure, only 50% right
    _, ece = calibration(probs, labels)
    assert ece == pytest.approx(0.4)


@pytest.mark.skipif(not (STANFORD / "file_list.mat").exists(), reason="Stanford Dogs not downloaded")
def test_stanford_breeds_match_imagenet():
    breeds = stanford_breeds(EFFNET_B0.categories)
    assert len(breeds) == 120
    assert len({b.imagenet_index for b in breeds}) == 120
    names = {b.name for b in breeds}
    assert "Cardigan" in names and "dingo" in names and "dalmatian" not in names
    assert next(b for b in breeds if b.name == "Cardigan").imagenet_index == 264   # the corgi, not the sweater


def test_report_file_is_valid():
    report = Path(__file__).resolve().parent.parent / "reports" / "stage4_report.json"
    assert report.exists(), "run scripts/stage4_report.py"
    data = json.loads(report.read_text())
    assert len(data["breeds"]) == 120
    for m in data["models"].values():
        r = m["stanford_test"]
        assert r["photos"] == 8580 and 0 < r["top1"] <= r["top3"] <= r["top5"] <= 1
