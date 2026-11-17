"""
Tests every predictor must pass: the "contract" of the Predictor interface.

When later stages add real models to PREDICTORS, these same tests run on them
automatically, so a new model can't silently break what the app expects.

Run with:  uv run pytest
"""
import pytest
from PIL import Image

from dogbreeds.predictors import PREDICTORS, DummyPredictor


@pytest.fixture
def image():
    return Image.new("RGB", (224, 224), color="brown")


@pytest.mark.parametrize("name", list(PREDICTORS))
@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_predictor_contract(name, top_k, image):
    predictions = PREDICTORS[name].predict(image, top_k=top_k)

    assert len(predictions) == top_k
    for breed, prob in predictions:
        assert isinstance(breed, str) and breed
        assert 0.0 <= prob <= 1.0
    probs = [p for _, p in predictions]
    assert probs == sorted(probs, reverse=True), "best guess must come first"
    assert sum(probs) <= 1.0 + 1e-6, "top-k probabilities can't add up to more than 1"
    assert len({b for b, _ in predictions}) == top_k, "no breed twice"


def test_dummy_is_repeatable_with_a_seed(image):
    assert DummyPredictor(seed=42).predict(image) == DummyPredictor(seed=42).predict(image)


# ---- Stage 2: the pre-trained ImageNet model ----
from dogbreeds.predictors import DOG_CLASSES, EFFNET_B0


def test_dog_classes_are_the_118_imagenet_breeds():
    names = [EFFNET_B0.categories[i] for i in DOG_CLASSES]
    assert len(names) == 118
    assert names[0] == "Chihuahua" and names[-1] == "Mexican hairless"
    assert EFFNET_B0.categories[DOG_CLASSES.start - 1] == "sea lion"   # just before: not a dog
    assert EFFNET_B0.categories[DOG_CLASSES.stop] == "timber wolf"     # just after: not a dog


def test_imagenet_analysis_is_consistent(image):
    a = EFFNET_B0.analyse(image, top_k=5)
    dog_names = {EFFNET_B0.categories[i] for i in DOG_CLASSES}
    assert all(breed in dog_names for breed, _ in a.breeds)
    assert 0.0 <= a.dog_share <= 1.0
    # the breeds' probabilities are part of the dog share, so they can't add up to more
    assert sum(p for _, p in a.breeds) <= a.dog_share + 1e-6
    assert a.seconds > 0
