"""
Stage 3: deciding what to tell the user, including "I don't know".

A model never refuses by itself: it always gives probabilities for the classes
it knows. We add the refusal on top, with two thresholds:

    dog share < dog_threshold            -> "no dog"         (red)
    breed confidence < breed_threshold   -> "a dog, unsure"  (yellow)
    otherwise                            -> "this breed"     (green)

The thresholds are not guesses: scripts/stage3_thresholds.py measures them on
photos of dogs, cats and food, and saves them in config/stage3_thresholds.json.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dogbreeds.predictors import ImageNetAnalysis, Prediction

THRESHOLDS_FILE = Path(__file__).resolve().parent.parent / "config" / "stage3_thresholds.json"


@dataclass
class Thresholds:
    dog: float     # minimum dog share to say "there is a dog"
    breed: float   # minimum breed confidence to name one breed


@dataclass
class Decision:
    status: Literal["breed", "unsure", "no_dog"]
    message: str
    breeds: Prediction   # breed probabilities given that it's a dog (they add up to at most 1)


def load_thresholds(model_name: str, path: Path = THRESHOLDS_FILE) -> Thresholds:
    """The measured thresholds for one model (each model has its own).

    Stage 5's model keeps its thresholds in config/stage5_thresholds.json (same format).
    """
    data = json.loads(path.read_text())
    t = data["models"][model_name]
    return Thresholds(dog=t["dog_threshold"], breed=t["breed_threshold"])


def breed_confidence(analysis: ImageNetAnalysis) -> Prediction:
    """Breed probabilities *given that it's a dog*: each breed's share of the dog share.

    Example: dog share 0.80 and beagle 0.60  ->  beagle is 0.60 / 0.80 = 75% of the dog part.
    This is a conditional probability: P(beagle | dog) = P(beagle) / P(dog).
    """
    share = max(analysis.dog_share, 1e-12)  # avoid dividing by zero
    return [(breed, p / share) for breed, p in analysis.breeds]


def decide(analysis: ImageNetAnalysis, thresholds: Thresholds) -> Decision:
    breeds = breed_confidence(analysis)

    if analysis.dog_share < thresholds.dog:
        thing, p = analysis.imagenet_top[0]
        return Decision("no_dog", f"🔴 **I don't see a dog.** This looks most like: *{thing}* ({p:.0%}).", breeds)

    top_breed, top_p = breeds[0]
    if top_p < thresholds.breed:
        names = ", ".join(b for b, _ in breeds[:2]) + f" or {breeds[2][0]}" if len(breeds) >= 3 \
            else " or ".join(b for b, _ in breeds)
        return Decision("unsure", f"🟡 **A dog, but I'm not sure which breed.** Maybe a {names}.", breeds)

    return Decision("breed", f"🟢 **{top_breed}** ({top_p:.0%} sure)", breeds)
