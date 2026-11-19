"""Stage 7: the heatmap maths, checked against the model itself and on tiny examples."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dogbreeds.heatmap import class_maps, feature_maps, heat_inside, hottest_on_pet, mask_view, model_view, to_heat
from dogbreeds.predictors import EFFNET_B0

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = EFFNET_B0.model
    x = torch.rand(2, 3, 224, 224)
    return model, x, feature_maps(model, x)


def test_the_average_of_a_class_map_is_the_class_score(setup):
    """The map is each cell's share of the score: averaged over the 9x9 grid (plus bias) it IS the logit."""
    model, x, A = setup
    classes = [151, 250, 948]
    cams = class_maps(model, A, classes)
    with torch.inference_mode():
        logits = model(x)
    bias = model.classifier[1].bias[classes]
    assert torch.allclose(cams.mean(dim=(2, 3)) + bias, logits[:, classes], atol=1e-3)


def test_cam_equals_grad_cam_for_this_model(setup):
    """Grad-CAM, computed the general way with autograd, gives the same map (up to a constant factor)."""
    model, x, _ = setup
    A = model.features(x[:1]).detach().requires_grad_(True)
    score = model.classifier(torch.flatten(model.avgpool(A), 1))[0, 250]
    score.backward()
    alpha = A.grad.mean(dim=(2, 3), keepdim=True)                 # Grad-CAM: average gradient per feature map
    grad_cam = (alpha * A).sum(dim=1)[0].detach()
    cam = class_maps(model, A.detach(), [250])[0, 0]
    assert torch.allclose(grad_cam * A.shape[-1] * A.shape[-2], cam, atol=1e-3)


def test_different_breeds_give_different_maps(setup):
    model, _, A = setup
    cams = class_maps(model, A, [250, 248])
    assert not torch.allclose(cams[:, 0], cams[:, 1])


def test_one_class_per_photo_matches_the_list_form(setup):
    model, _, A = setup
    per_photo = class_maps(model, A, torch.tensor([250, 248]))
    as_list = class_maps(model, A, [250, 248])
    assert torch.allclose(per_photo[0, 0], as_list[0, 0]) and torch.allclose(per_photo[1, 0], as_list[1, 1])


def test_to_heat_is_between_0_and_1():
    heat = to_heat(torch.tensor([[-1.0, 2.0], [0.5, 4.0]]), size=8)
    assert heat.shape == (8, 8) and heat.min() == 0 and heat.max() == pytest.approx(1.0)
    assert to_heat(-torch.ones(2, 2), size=4).max() == 0          # no evidence anywhere: all zero


def test_heat_inside_and_pointing_game():
    mask = np.zeros((4, 4), bool)
    mask[:2, :2] = True                                           # the "pet" is the top-left quarter
    heat = np.zeros((4, 4))
    heat[0, 0], heat[3, 3] = 1.0, 0.25
    assert heat_inside(heat, mask) == pytest.approx(0.8)
    assert hottest_on_pet(heat, mask)


def test_photo_and_outline_are_cut_out_the_same_way():
    photo = Image.new("RGB", (400, 300))
    outline = Image.new("L", (400, 300), 1)
    x, crop = model_view(photo, EFFNET_B0.weights)
    mask = mask_view(outline, EFFNET_B0.weights)
    assert x.shape == (1, 3, 224, 224) and crop.size == (224, 224) and mask.shape == (224, 224) and mask.all()


@pytest.mark.skipif(not (ROOT / "reports" / "stage7_report.json").exists(), reason="run scripts/stage7_heatmaps.py")
def test_stage7_report_is_valid():
    r = json.loads((ROOT / "reports" / "stage7_report.json").read_text())
    for group in r["groups"].values():
        assert 0 <= group["pointing_game"] <= 1 and 0 <= group["heat_on_pet"] <= 1 and 0 < group["pet_area"] < 1


def test_fix_corner_only_changes_the_top_left_cell():
    from dogbreeds.heatmap import fix_corner
    A = torch.arange(2 * 3 * 9 * 9, dtype=torch.float32).reshape(2, 3, 9, 9)
    A[..., 0, 0] = 1000.0
    fixed = fix_corner(A)
    assert torch.equal(fixed[..., 1:, :], A[..., 1:, :]) and torch.equal(fixed[..., 0, 1:], A[..., 0, 1:])
    assert torch.allclose(fixed[..., 0, 0], (A[..., 0, 1] + A[..., 1, 0] + A[..., 1, 1]) / 3)


def test_the_corner_artifact_is_in_the_model_not_the_photo():
    """Even on a plain grey image, the top-left cell is far stronger than the rest."""
    from dogbreeds.predictors import EFFNET_B2
    x, _ = model_view(Image.new("RGB", (300, 300), (128, 128, 128)), EFFNET_B2.weights)
    A = feature_maps(EFFNET_B2.model, x)[0].abs().mean(dim=0)       # average strength per cell
    assert A[0, 0] > 20 * A[1:, 1:].mean()
