"""Stage 5: the training pieces, tested on small made-up data where the right answer is known."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dogbreeds.finetune import BreedHead, backbone, head_from_imagenet, predict_probs, stratified_split, train_head
from dogbreeds.predictors import EFFNET_B0, EFFNET_B2, FINETUNED_B2, HEAD_FILE

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


# ---- Stage 6: partly frozen fine-tuning ----
import torchvision

from dogbreeds.finetune import PartlyFrozen, train_transform
from dogbreeds.predictors import FINETUNED6_B2, STAGE6_FILE


def small_partly_frozen(first_trained_block=7):
    """An untrained EfficientNet-B0 (no download needed) with 3 breeds, for fast tests."""
    torch.manual_seed(0)
    base = torchvision.models.efficientnet_b0(weights=None).eval()
    return base, PartlyFrozen(base, first_trained_block, head_from_imagenet(base.classifier[1], [151, 200, 264]))


def test_augmentation_gives_a_different_picture_each_time():
    t = train_transform(EFFNET_B0.weights)
    photo = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    a, b = t(photo), t(photo)
    assert a.shape == (3, 224, 224)            # the model's own input size (B0: 224, B2: 288)
    assert not torch.equal(a, b)


def test_partly_frozen_starts_as_the_imagenet_model():
    base, model = small_partly_frozen()
    x = torch.rand(2, 3, 224, 224)
    with torch.inference_mode():
        p = torch.softmax(base(x), dim=1)[:, [151, 200, 264]]
        assert torch.allclose(torch.softmax(model(x), dim=1), p / p.sum(dim=1, keepdim=True), atol=1e-5)


def test_training_changes_only_the_unfrozen_part():
    _, model = small_partly_frozen()
    frozen_before = [p.clone() for p in model.frozen.parameters()]
    tuned_before = [p.clone() for p in model.tuned.parameters()]
    model.train()
    assert not model.tuned.training and model.head.training   # BatchNorm stays in eval mode, dropout is on
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss = torch.nn.functional.cross_entropy(model(torch.rand(4, 3, 224, 224)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(frozen_before, model.frozen.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(tuned_before, model.tuned.parameters()))


needs_stage6 = pytest.mark.skipif(not (ROOT / "reports" / "stage6_report.json").exists(),
                                  reason="run scripts/stage6_finetune.py")


@needs_stage6
def test_stage6_shares_the_frozen_blocks_and_keeps_imagenets_dog_check():
    image = Image.new("RGB", (300, 300), "brown")
    result = FINETUNED6_B2.compare(image, top_k=3)
    assert result["imagenet"].dog_share == pytest.approx(EFFNET_B2.analyse(image).dog_share, abs=1e-5)
    assert result["ours"].dog_share == result["imagenet"].dog_share
    assert len(FINETUNED6_B2.breeds) == 120


@needs_stage6
def test_stage6_report_is_valid():
    report = json.loads((ROOT / "reports" / "stage6_report.json").read_text())
    assert set(report["test"]) == {"imagenet_head", "stage5_head", "stage6"}
    assert report["history"][0]["val_accuracy"] == pytest.approx(0.9467, abs=0.002)   # epoch 0 = ImageNet's head
    assert isinstance(report["ship_rule"]["ship"], bool)
