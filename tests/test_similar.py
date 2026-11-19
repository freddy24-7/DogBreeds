"""Stage 8: the similarity search, checked on small made-up embeddings."""
import json
from pathlib import Path

import numpy as np
import pytest

from dogbreeds.similar import Index, normalise, same_breed_share, vote

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def gallery():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 32)) * 3                         # 3 "breeds", each a cloud of points
    labels = np.repeat(np.arange(3), 20)
    feats = (centres[labels] + rng.normal(size=(60, 32))).astype(np.float32)
    return feats, labels, [f"photo_{i}.jpg" for i in range(60)]


def test_normalise_gives_length_one():
    x = normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(x[0]), 1) and np.all(x[1] == 0)


@pytest.mark.parametrize("dims", [None, 8])
def test_a_gallery_photo_finds_itself_first(gallery, dims):
    feats, labels, paths = gallery
    index = Index.build(feats, labels, paths, dims=dims)
    sims, pos = index.search(feats, k=3)
    assert np.all(pos[:, 0] == np.arange(60)) and np.allclose(sims[:, 0], 1, atol=1e-5)
    assert np.all(np.diff(sims, axis=1) <= 1e-6)                   # best first
    assert index.dims == (dims or 32) and np.allclose(np.linalg.norm(index.vectors, axis=1), 1, atol=1e-5)


def test_skip_above_leaves_out_the_photo_itself(gallery):
    feats, labels, paths = gallery
    _, pos = Index.build(feats, labels, paths).search(feats, k=3, skip_above=0.999)
    assert not np.any(pos == np.arange(60)[:, None])


def test_searching_in_chunks_gives_the_same_answer(gallery):
    feats, labels, paths = gallery
    index = Index.build(feats, labels, paths)
    a, b = index.search(feats, k=5), index.search(feats, k=5, chunk=7)
    assert np.allclose(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_neighbours_share_the_breed_on_clear_clusters(gallery):
    feats, labels, paths = gallery
    index = Index.build(feats, labels, paths)
    sims, pos = index.search(feats, k=6, skip_above=0.999)
    assert same_breed_share(labels[pos], labels).mean() > 0.95
    assert (vote(sims, labels[pos], 3).argmax(1) == labels).mean() > 0.95


def test_same_breed_share_and_vote_on_a_tiny_example():
    neighbours = np.array([[1, 1, 2, 1]])
    assert same_breed_share(neighbours, [1])[0] == 0.75
    shares = vote(np.array([[0.9, 0.5, 0.8, -0.2]]), neighbours, 3)   # negative similarity: no vote
    assert np.allclose(shares[0], [0, 1.4 / 2.2, 0.8 / 2.2])


def test_save_and_load_give_the_same_search(gallery, tmp_path):
    feats, labels, paths = gallery
    index = Index.build(feats, labels, paths, dims=8)
    index.save(tmp_path / "index.npz")
    loaded = Index.load(tmp_path / "index.npz")
    assert loaded.paths == paths and np.array_equal(loaded.labels, labels)
    assert np.array_equal(loaded.search(feats, k=3)[1][:, 0], index.search(feats, k=3)[1][:, 0])


@pytest.mark.skipif(not (ROOT / "reports" / "stage8_report.json").exists(), reason="run scripts/stage8_similar.py")
def test_stage8_report_is_valid():
    r = json.loads((ROOT / "reports" / "stage8_report.json").read_text())
    assert r["chosen"] in {v["key"] for v in r["variants"]}
    for s in r["test"].values():
        assert 0 < s["precision_at_5"] <= 1 and 0 < s["knn_accuracy"] <= 1
    assert r["copies"]["copies"] <= r["copies"]["same_shoot"] <= r["copies"]["test_photos"]
