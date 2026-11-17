"""Stage 5: the training pieces, tested on small made-up data where the right answer is known."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dogbreeds.finetune import BreedHead, backbone, head_from_imagenet, predict_probs, stratified_split, train_head
from dogbreeds.predictors import EFFNET_B0, FINETUNED_B2, HEAD_FILE

ROOT = Path(__file__).resolve().parent.parent


def test_backbone_plus_imagenet_head_is_the_whole_model():
    """Splitting the model in two must not change its output."""
    model = EFFNET_B0.model
    x = torch.rand(2, 3, 224, 224)
    with torch.inference_mode():
        assert torch.allclose(model.classifier(backbone(model, x)), model(x), atol=1e-4)
        assert backbone(model, x).shape == (2, 1280)   # B0 has 1,280 features; B2 has 1,408


def test_head_from_imagenet_starts_with_the_stage4_predictions():
    """Softmax over 120 of the 1000 scores = the 1000 probabilities with 120 kept and rescaled."""
    torch.manual_seed(0)
    imagenet = torch.nn.Linear(16, 1000)
    keep = [151, 200, 264, 273]
    features = torch.randn(5, 16)
    ours = predict_probs(head_from_imagenet(imagenet, keep), features)
    with torch.no_grad():
        p = torch.softmax(imagenet(features), dim=1)[:, keep]
    assert np.allclose(ours, (p / p.sum(dim=1, keepdim=True)).numpy(), atol=1e-6)


def test_stratified_split_takes_the_same_share_of_every_breed():
    labels = np.repeat(np.arange(4), 100)          # 4 breeds, 100 photos each
    train, val = stratified_split(labels, val_share=0.1)
    assert len(set(train) & set(val)) == 0, "a photo can't be in both"
    assert len(train) + len(val) == 400
    assert np.bincount(labels[val]).tolist() == [10, 10, 10, 10]
    assert stratified_split(labels)[1].tolist() == val.tolist(), "same seed, same split"


def test_training_learns_an_easy_problem():
    """Two clearly separated groups of points: training must lower the loss and get them right."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 1, (100, 8)), rng.normal(2, 1, (100, 8))]).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100)
    head = BreedHead(in_features=8, n_breeds=2, dropout=0.0)
    state, history, best_epoch = train_head(head, x[::2], y[::2], x[1::2], y[1::2], epochs=10, lr=0.05)
    assert history[best_epoch]["val_loss"] < history[0]["val_loss"]
    assert history[best_epoch]["val_accuracy"] > 0.95
    head.load_state_dict(state)
    assert predict_probs(head, x).shape == (200, 2)


needs_head = pytest.mark.skipif(not HEAD_FILE.exists(), reason="run scripts/stage5_train_head.py")


@needs_head
def test_both_heads_share_one_backbone_pass():
    result = FINETUNED_B2.compare(Image.new("RGB", (300, 300), "brown"), top_k=3)
    theirs, ours = result["imagenet"], result["ours"]
    assert theirs.dog_share == ours.dog_share          # "is it a dog?" still comes from ImageNet's head
    assert len(FINETUNED_B2.breeds) == 120
    assert sum(p for _, p in ours.breeds) <= ours.dog_share + 1e-6   # same format as stage 3


@needs_head
def test_stage5_report_is_valid():
    report = json.loads((ROOT / "reports" / "stage5_report.json").read_text())
    assert report["split"] == {"train": 10800, "validation": 1200, "test": 8580}
    assert report["cache_check"]["same_top1_share"] > 0.995
    for model in report["test"].values():
        assert model["stanford_test"]["photos"] == 8580 and model["oxford_test"]["photos"] == 2086
