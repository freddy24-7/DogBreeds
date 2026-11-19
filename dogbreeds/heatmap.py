"""
Stage 7: where did the model look? Class activation maps (CAM / Grad-CAM).

The last part of EfficientNet's backbone gives 1,408 feature maps of 9 x 9 cells:
for every cell of a 9 x 9 grid laid over the photo, 1,408 numbers saying what's there.

    photo [3, 288, 288] -> backbone -> A [1408, 9, 9] -> average over the 9x9 grid -> 1,408 numbers -> head

The head's score for a breed is a weighted sum of those 1,408 averages:

    score(breed) = sum_c  w[breed, c] * mean_over_cells(A[c])  + bias
                 = mean_over_cells( sum_c w[breed, c] * A[c] ) + bias

So the score is the AVERAGE of a 9 x 9 map, sum_c w[breed, c] * A[c]: each cell's share of the
score. That map is the class activation map (CAM). High cells pushed the score up: "this is
where it saw the breed". We blow it up from 9 x 9 to the photo size and lay it over the photo.

Grad-CAM, the more general method, weighs each feature map by the gradient of the score.
For this model that gradient is w[breed, c] / 81 in every cell, so Grad-CAM gives the same
map (tests/test_heatmap.py checks it). CAM just doesn't need a backward pass.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ORANGE = np.array([0xEB, 0x68, 0x34], dtype=np.float32)   # validated palette, slot 2


def model_view(image: Image.Image, weights) -> tuple[torch.Tensor, Image.Image]:
    """The photo exactly as the model sees it: (normalised tensor [1, 3, S, S], the S x S crop as a picture).

    The model's transform resizes the short side to S and cuts out the centre square.
    Anything outside that square, the model never sees.
    """
    t = weights.transforms()
    size = t.crop_size[0]
    crop = TF.center_crop(TF.resize(image.convert("RGB"), t.resize_size[0], interpolation=t.interpolation), size)
    x = TF.normalize(TF.to_tensor(crop), t.mean, t.std).unsqueeze(0)
    return x, crop


def mask_view(mask: Image.Image, weights) -> np.ndarray:
    """An Oxford outline (trimap) cut out exactly like the photo. True = pet (or its border)."""
    t = weights.transforms()
    m = TF.center_crop(TF.resize(mask, t.resize_size[0], interpolation=TF.InterpolationMode.NEAREST), t.crop_size[0])
    return np.isin(np.array(m), [1, 3])     # trimap values: 1 pet, 2 background, 3 border


def feature_maps(model, x: torch.Tensor) -> torch.Tensor:
    """The backbone's output before averaging: [N, 1408, 9, 9]."""
    with torch.inference_mode():
        return model.features(x)


def fix_corner(A: torch.Tensor) -> torch.Tensor:
    """Repairs the top-left cell of feature maps (or class maps): [..., 9, 9] -> same shape.

    This model's top-left cell is about 280 times stronger than the others, on any photo, even a
    plain grey one, and it stays top-left when the photo is mirrored. So it's an artifact of the
    model, not something in the photo. Most likely cause: the weights were trained in TensorFlow,
    which pads image edges unevenly, and are used here in PyTorch, which pads evenly.
    We replace that cell with the average of its three neighbours.
    """
    fixed = A.clone()
    fixed[..., 0, 0] = (A[..., 0, 1] + A[..., 1, 0] + A[..., 1, 1]) / 3
    return fixed


def class_maps(model, A: torch.Tensor, classes) -> torch.Tensor:
    """The raw CAM for each requested ImageNet class: [N, len(classes), 9, 9].

    classes: a list of class numbers (the same for every photo), or a tensor [N] (one per photo).
    The average of a raw map plus the head's bias equals that class's score (logit).
    """
    W = model.classifier[1].weight                                  # [1000, 1408]
    with torch.inference_mode():
        if isinstance(classes, torch.Tensor):                       # one class per photo
            return torch.einsum("nc,nchw->nhw", W[classes], A).unsqueeze(1)
        return torch.einsum("kc,nchw->nkhw", W[list(classes)], A)


def to_heat(cam: torch.Tensor, size: int) -> np.ndarray:
    """A raw 9 x 9 map -> [size, size] heat between 0 and 1.

    Smoothly resized to the photo size; negative cells (evidence AGAINST the class) are set
    to 0; then scaled so the hottest spot is 1. A map that is nowhere positive stays all 0.
    """
    up = F.interpolate(cam.reshape(1, 1, *cam.shape[-2:]).float(), size=(size, size), mode="bilinear",
                       align_corners=False)[0, 0].clamp_min(0)
    top = up.max()
    return (up / top).numpy() if top > 0 else up.numpy()


def overlay(crop: Image.Image, heat: np.ndarray) -> Image.Image:
    """Lays the heat over the photo: orange where the model looked, the rest slightly dimmed."""
    img = np.asarray(crop, dtype=np.float32)
    h = heat[..., None]
    out = img * (0.45 + 0.55 * h)                                   # dim where it didn't look
    out = out * (1 - 0.55 * h) + ORANGE * 0.55 * h                  # orange where it did
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def heat_inside(heat: np.ndarray, mask: np.ndarray) -> float:
    """Share of the heat that falls on the pet (0 to 1). Compare with the pet's share of the picture."""
    total = heat.sum()
    return float((heat * mask).sum() / total) if total > 0 else float("nan")


def hottest_on_pet(heat: np.ndarray, mask: np.ndarray) -> bool:
    """The 'pointing game': is the single hottest spot on the pet?"""
    return bool(mask.flat[int(heat.argmax())])
