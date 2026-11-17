"""
Stage 4: measuring a model properly. Every metric works on the same two arrays:

    probs   [N, 120]  the model's breed probabilities for N photos
    labels  [N]       the true breed number (0..119) of each photo

So any model (ImageNet now, fine-tuned in stage 5) can be compared on exactly
the same footing: produce `probs` for the test photos, then call `model_report`.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dogbreeds.data import Breed, PhotoDataset


# ---------------------------------------------------------------- predictions

def imagenet_breed_probs(predictor, dataset: PhotoDataset, breeds: list[Breed], device: str,
                         batch_size: int = 32, num_workers: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Runs an ImageNet model on a dataset and returns (probs, labels, seconds per photo).

    probs are over the 120 Stanford breeds and add up to 1 per photo:
    softmax over all 1000 ImageNet classes, keep the 120 breed classes, rescale.
    That's the breed confidence from stage 3: "if it's a dog, which breed?".
    """
    dataset.transform = predictor.weights.transforms()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    cols = torch.tensor([b.imagenet_index for b in breeds])
    model = predictor.model.to(device)
    all_probs, all_labels = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1).cpu()[:, cols]
            all_probs.append(p / p.sum(dim=1, keepdim=True))
            all_labels.append(y)
    seconds = (time.perf_counter() - start) / len(dataset)
    predictor.model.to("cpu")
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy(), seconds


# ---------------------------------------------------------------- metrics

def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Share of photos whose true breed is among the k most likely."""
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([label in row for row, label in zip(topk, labels)]))


def per_breed_accuracy(probs: np.ndarray, labels: np.ndarray, n_breeds: int) -> np.ndarray:
    """Top-1 accuracy for each breed separately (NaN for a breed with no photos)."""
    correct = probs.argmax(axis=1) == labels
    return np.array([correct[labels == b].mean() if (labels == b).any() else np.nan for b in range(n_breeds)])


def confusion_matrix(probs: np.ndarray, labels: np.ndarray, n_breeds: int) -> np.ndarray:
    """cm[true, predicted] = number of photos. The diagonal holds the correct ones."""
    cm = np.zeros((n_breeds, n_breeds), dtype=int)
    np.add.at(cm, (labels, probs.argmax(axis=1)), 1)
    return cm


def confused_pairs(cm: np.ndarray, top: int = 10) -> list[tuple[int, int, int]]:
    """The most frequent mistakes as (true breed, predicted breed, count)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:top]
    return [(int(t), int(p), int(off[t, p])) for t, p in zip(*np.unravel_index(flat, off.shape)) if off[t, p] > 0]


def calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[list[dict], float]:
    """Is "80% sure" right 80% of the time?

    Photos are grouped by the model's confidence (its top probability). For each
    group we compare the average confidence with the actual accuracy. The
    expected calibration error (ECE) is the average gap, weighted by group size:
    0 = perfectly calibrated.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            bins.append(dict(low=float(lo), high=float(hi), count=int(mask.sum()),
                             confidence=float(confidence[mask].mean()), accuracy=float(correct[mask].mean())))
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return bins, float(ece)


def model_report(probs: np.ndarray, labels: np.ndarray, breeds: list[Breed]) -> dict:
    """All stage 4 metrics for one model on one test set, as plain JSON-friendly values."""
    n = len(breeds)
    per_breed = per_breed_accuracy(probs, labels, n)
    order = np.argsort(per_breed)
    cm = confusion_matrix(probs, labels, n)
    bins, ece = calibration(probs, labels)
    return {
        "photos": int(len(labels)),
        "top1": topk_accuracy(probs, labels, 1),
        "top3": topk_accuracy(probs, labels, 3),
        "top5": topk_accuracy(probs, labels, 5),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "ece": ece,
        "calibration": bins,
        "per_breed": {breeds[b].name: round(float(a), 4) for b, a in enumerate(per_breed) if not np.isnan(a)},
        "hardest": [(breeds[b].name, round(float(per_breed[b]), 4)) for b in order[:10]],
        "easiest": [(breeds[b].name, round(float(per_breed[b]), 4)) for b in order[::-1][:10]
                    if not np.isnan(per_breed[b])],
        "confused_pairs": [dict(true=breeds[t].name, predicted=breeds[p].name, count=c,
                                share=round(c / max(int((labels == t).sum()), 1), 4))
                           for t, p, c in confused_pairs(cm)],
    }
