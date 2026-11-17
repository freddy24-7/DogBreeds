"""
Predictors: the "model" part of the app, hidden behind one small interface.

The app only ever calls `predictor.predict(image)` and gets back a list of
(breed, probability) pairs, best first. It doesn't know or care what happens
inside. That lets us start with a fake predictor (stage 1) and swap in real
models later (stage 2: pre-trained ImageNet model, stage 5: fine-tuned model)
without changing the UI. Same idea as mocking a service in normal software.
"""
import random
import time
from dataclasses import dataclass
from typing import Protocol

import torch
import torchvision
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


# --------------------------------------------------------------------------
# Stage 2: a real model, pre-trained on ImageNet. No training by us.
# --------------------------------------------------------------------------

# ImageNet has 1000 classes. Numbers 151 up to and including 268 are dog
# breeds (118 of them, from "Chihuahua" to "Mexican hairless").
DOG_CLASSES = range(151, 269)


@dataclass
class ImageNetAnalysis:
    """Everything the ImageNet model says about one photo (shown in the stage 2 tab)."""

    breeds: Prediction           # top dog breeds, with their probability out of all 1000 classes
    dog_share: float             # how much of the total probability went to dog breeds (0 to 1)
    imagenet_top: Prediction     # top classes out of ALL 1000, dog or not
    seconds: float               # how long the prediction took


class ImageNetPredictor:
    """Stage 2: a pre-trained ImageNet model, used as it is.

    How it works, in four steps:
      1. Prepare the photo exactly like the model's training photos
         (resize, crop, normalise): the transform that belongs to the weights.
      2. The model gives 1000 scores ("logits"), one per ImageNet class.
      3. Softmax turns the scores into probabilities that add up to 1.
      4. We keep only the 118 dog breeds and return the best ones.

    We do NOT rescale the dog probabilities to add up to 1 again. So for a
    photo of a cat, every breed gets a low probability. That's a useful
    warning sign, which stage 3 turns into "I don't see a dog".
    """

    def __init__(self, name: str, builder, weights):
        self.name = name
        self._builder = builder          # e.g. torchvision.models.efficientnet_b0
        self.weights = weights           # e.g. EfficientNet_B0_Weights.IMAGENET1K_V1
        self.categories: list[str] = weights.meta["categories"]
        self._model = None               # loaded on first use, so the app starts quickly

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            # The weights file (20-40 MB) is downloaded once, then read from the local cache.
            # eval(): switch off training-only behaviour such as dropout.
            self._model = self._builder(weights=self.weights).eval()
        return self._model

    def analyse(self, image: Image.Image, top_k: int = 3) -> ImageNetAnalysis:
        start = time.perf_counter()
        x = self.weights.transforms()(image.convert("RGB")).unsqueeze(0)  # [1, 3, H, W]: a batch of one
        with torch.inference_mode():                                       # no gradients needed: faster
            logits = self.model(x)[0]                                      # 1000 scores
        probs = torch.softmax(logits, dim=0)                               # 1000 probabilities, sum = 1

        dog_probs = probs[DOG_CLASSES.start:DOG_CLASSES.stop]
        top_dogs = dog_probs.topk(top_k)
        breeds = [(self.categories[DOG_CLASSES.start + i], p)
                  for p, i in zip(top_dogs.values.tolist(), top_dogs.indices.tolist())]

        top_all = probs.topk(top_k)
        imagenet_top = [(self.categories[i], p) for p, i in zip(top_all.values.tolist(), top_all.indices.tolist())]

        return ImageNetAnalysis(breeds=breeds, dog_share=dog_probs.sum().item(),
                                imagenet_top=imagenet_top, seconds=time.perf_counter() - start)

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        return self.analyse(image, top_k).breeds


EFFNET_B0 = ImageNetPredictor("EfficientNet-B0 (ImageNet, 5.3M parameters)",
                              torchvision.models.efficientnet_b0,
                              torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
EFFNET_B2 = ImageNetPredictor("EfficientNet-B2 (ImageNet, 9.1M parameters)",
                              torchvision.models.efficientnet_b2,
                              torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)

# All predictors in the app. Each stage's tab picks the ones it shows.
DUMMY = DummyPredictor()
PREDICTORS: dict[str, Predictor] = {p.name: p for p in [DUMMY, EFFNET_B0, EFFNET_B2]}
