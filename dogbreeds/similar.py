"""
Stage 8: dogs that look like yours. Similarity search with embeddings.

The backbone turns every photo into 1,408 numbers (stage 5). Until now a head turned
those numbers into breed scores. Here we use the numbers directly, as a point in a
1,408-dimensional space: an EMBEDDING. Photos that look alike land close together.

    gallery (12,000 Stanford training photos) --backbone--> 12,000 embeddings  (done once, saved)
    your photo --backbone--> 1 embedding --compare with all 12,000--> the closest ones

"Close" is measured with COSINE SIMILARITY: the angle between two embeddings, ignoring
their length. 1 = same direction, 0 = unrelated. Two steps make it work better:

  1. CENTRE: subtract the gallery's average embedding. Every photo shares a big
     "photo of a dog" component; removing it leaves what makes each photo different.
  2. NORMALISE: scale every embedding to length 1. Then cosine similarity is a plain
     dot product, and comparing with all 12,000 photos is one matrix multiplication.

Optionally, PCA first keeps only the most varied directions (fewer numbers per photo:
a smaller index, a faster search). This is exactly what a vector database does, at a
small scale and without the tricks it needs for millions of items.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dogbreeds.finetune import backbone


def embed(predictor, images) -> np.ndarray:
    """Photos (PIL) -> raw embeddings [N, 1408], prepared exactly like the model's training photos."""
    t = predictor.weights.transforms()
    x = torch.stack([t(im.convert("RGB")) for im in images])
    with torch.inference_mode():
        return backbone(predictor.model, x).numpy()


def normalise(x: np.ndarray) -> np.ndarray:
    """Every row scaled to length 1 (rows of zeros stay zero)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


@dataclass
class Index:
    """A searchable gallery: its embeddings (centred, optionally PCA'd, normalised) plus labels and photo paths."""

    mean: np.ndarray                 # [1408] the gallery's average embedding
    components: np.ndarray | None    # [dims, 1408] PCA directions, or None for all 1,408 numbers
    vectors: np.ndarray              # [N, dims] ready to compare: length 1
    labels: np.ndarray               # [N] breed number of each gallery photo
    paths: list[str]                 # [N] where each gallery photo is (relative to the project folder)

    @classmethod
    def build(cls, feats: np.ndarray, labels, paths, dims: int | None = None, centre: bool = True) -> "Index":
        mean = feats.mean(axis=0) if centre else np.zeros(feats.shape[1], feats.dtype)
        components = None
        if dims is not None:
            # PCA: the directions along which the (centred) gallery varies most, from the SVD
            _, _, vt = np.linalg.svd(feats - mean, full_matrices=False)
            components = vt[:dims]
        index = cls(mean.astype(np.float32), components, np.empty(0), np.asarray(labels), list(paths))
        index.vectors = index.project(feats)
        return index

    @property
    def dims(self) -> int:
        return self.vectors.shape[1]

    def project(self, feats: np.ndarray) -> np.ndarray:
        """Raw embeddings [N, 1408] -> this index's space [N, dims]: centre, (PCA), length 1."""
        x = feats - self.mean
        if self.components is not None:
            x = x @ self.components.T
        return normalise(x).astype(np.float32)

    def search(self, feats: np.ndarray, k: int = 8, skip_above: float | None = None, chunk: int = 1024):
        """The k most similar gallery photos for each query: (similarities [Q, k], gallery positions [Q, k]).

        skip_above: leave out gallery photos at least this similar (copies of the query, or the same photo shoot).
        Queries are compared in chunks, so 8,580 queries x 12,000 photos never sit in memory at once.
        """
        q = self.project(feats)
        all_sims, all_top = [], []
        for start in range(0, len(q), chunk):
            sims = q[start:start + chunk] @ self.vectors.T        # cosine similarity with every gallery photo
            if skip_above is not None:
                sims = np.where(sims >= skip_above, -np.inf, sims)
            top = np.argpartition(-sims, k, axis=1)[:, :k]        # the k best, in no order (fast)
            order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)          # sorted best first
            all_sims.append(np.take_along_axis(sims, top, axis=1))
            all_top.append(top)
        return np.concatenate(all_sims), np.concatenate(all_top)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, vectors=self.vectors.astype(np.float16), labels=self.labels,
                 paths=np.array(self.paths), **({} if self.components is None else {"components": self.components}))

    @classmethod
    def load(cls, path: Path) -> "Index":
        z = np.load(path)
        return cls(z["mean"], z["components"] if "components" in z.files else None,
                   z["vectors"].astype(np.float32), z["labels"], [str(p) for p in z["paths"]])


def same_breed_share(neighbour_labels: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    """Precision@k per query: the share of its k neighbours that have the query's breed. [Q, k] -> [Q]."""
    return (neighbour_labels == np.asarray(query_labels)[:, None]).mean(axis=1)


def vote(sims: np.ndarray, neighbour_labels: np.ndarray, n_breeds: int) -> np.ndarray:
    """k-nearest-neighbour classifier: each neighbour votes for its breed, weighted by its similarity.

    Returns [Q, n_breeds] vote shares (rows sum to 1). No training at all: the 'model' is the gallery.
    """
    w = np.clip(sims, 0, None)
    votes = np.zeros((len(sims), n_breeds))
    np.add.at(votes, (np.arange(len(sims))[:, None], neighbour_labels), w)
    total = votes.sum(axis=1, keepdims=True)
    return np.divide(votes, total, out=np.zeros_like(votes), where=total > 0)
