"""
Predictors: the "model" part of the app, hidden behind one small interface.

The app only ever calls `predictor.predict(image)` and gets back a list of
(breed, probability) pairs, best first. It doesn't know or care what happens
inside. That lets us start with a fake predictor (stage 1) and swap in real
models later (stage 2: pre-trained ImageNet model, stage 5: fine-tuned model)
without changing the UI. Same idea as mocking a service in normal software.
"""
import random
from typing import Protocol

from PIL import Image

# A prediction: [("beagle", 0.82), ("basset", 0.09), ...], sorted best first.
Prediction = list[tuple[str, float]]


class Predictor(Protocol):
    """What every predictor must offer. Anything with these two attributes fits."""

    name: str  # shown in the app's model picker

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        """Return the top_k most likely breeds with their probabilities, best first."""
        ...


# A short list of common breeds, only used by the fake predictor.
# From stage 2 on, the breed names come from the real model.
DUMMY_BREEDS = [
    "Labrador retriever", "golden retriever", "German shepherd", "beagle", "pug",
    "French bulldog", "border collie", "Siberian husky", "dachshund", "poodle",
    "Chihuahua", "Yorkshire terrier", "boxer", "Dalmatian", "Shih-Tzu",
]


class DummyPredictor:
    """Stage 1: returns random breeds. Doesn't look at the image at all.

    It exists so we can build and test the app before there is any model.
    The probabilities are random but shaped like a real model's output:
    they're between 0 and 1, sorted, and sum to at most 1.
    """

    name = "Dummy (random guesses)"

    def __init__(self, seed: int | None = None):
        # A seed makes the "random" guesses repeatable, which is handy in tests.
        self.rng = random.Random(seed)

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        breeds = self.rng.sample(DUMMY_BREEDS, k=len(DUMMY_BREEDS))
        # Random positive scores, turned into probabilities that sum to 1
        # over ALL breeds (like softmax does), then keep only the top_k.
        scores = [self.rng.random() ** 3 for _ in breeds]  # **3 makes one breed stand out more
        total = sum(scores)
        ranked = sorted(zip(breeds, (s / total for s in scores)), key=lambda p: p[1], reverse=True)
        return ranked[:top_k]


# All predictors the app can choose from. New stages add entries here.
PREDICTORS: dict[str, Predictor] = {p.name: p for p in [DummyPredictor()]}
