"""Stage 3: the decision rules, tested with made-up analyses (no model needed)."""
import pytest

from dogbreeds.decision import THRESHOLDS_FILE, Thresholds, breed_confidence, decide, load_thresholds
from dogbreeds.predictors import EFFNET_B0, EFFNET_B2, ImageNetAnalysis

T = Thresholds(dog=0.5, breed=0.6)


def analysis(dog_share, breeds, top="tabby cat"):
    return ImageNetAnalysis(breeds=breeds, dog_share=dog_share, imagenet_top=[(top, 0.7)], seconds=0.1)


def test_breed_confidence_is_share_of_the_dog_part():
    a = analysis(0.8, [("beagle", 0.6), ("basset", 0.1), ("harrier", 0.05)])
    assert breed_confidence(a)[0] == ("beagle", pytest.approx(0.75))


def test_low_dog_share_means_no_dog():
    d = decide(analysis(0.1, [("Chihuahua", 0.05), ("pug", 0.02), ("boxer", 0.01)]), T)
    assert d.status == "no_dog" and "tabby cat" in d.message


def test_dog_with_low_breed_confidence_is_unsure():
    d = decide(analysis(0.9, [("beagle", 0.4), ("harrier", 0.3), ("basset", 0.1)]), T)
    assert d.status == "unsure" and "beagle" in d.message and "harrier" in d.message


def test_clear_dog_and_breed_is_named():
    d = decide(analysis(0.95, [("pug", 0.9), ("boxer", 0.02), ("bull mastiff", 0.01)]), T)
    assert d.status == "breed" and "pug" in d.message


def test_thresholds_exist_for_every_model_and_are_valid():
    assert THRESHOLDS_FILE.exists(), "run scripts/stage3_thresholds.py"
    for p in (EFFNET_B0, EFFNET_B2):
        t = load_thresholds(p.name)
        assert 0 < t.dog < 1 and 0 <= t.breed < 1
