"""
Stage 7: does the model look at the dog? Heatmaps measured against Oxford's pet outlines.

Run from the project folder (a few minutes):
    uv run python scripts/stage7_heatmaps.py

For every Oxford-IIIT Pets test photo (dogs and cats) it:
  1. runs EfficientNet-B2 (the model the app ships) and takes its answer: the top dog breed
     for dogs (as in stage 3), the top ImageNet class for cats;
  2. makes the heatmap for that answer;
  3. compares it with the photo's outline (Oxford "trimap": pet / border / background):
       - heat on pet:   the share of the heat that falls on the pet
       - pet area:      the pet's share of the picture (what a heatmap spread evenly would score)
       - pointing game: is the single hottest spot on the pet?
Writes reports/stage7_report.json, docs/stage7_heat_on_pet.png and example pictures in docs/stage7/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import stage3_thresholds as s3  # noqa: E402
from dogbreeds.heatmap import class_maps, feature_maps, fix_corner, heat_inside, hottest_on_pet, mask_view, model_view, overlay, to_heat  # noqa: E402
from dogbreeds.predictors import DOG_CLASSES, EFFNET_B2  # noqa: E402

PETS = ROOT / "data" / "oxford-iiit-pet"
OUT = ROOT / "docs" / "stage7"
COLORS = {"dog, right breed": "#2a78d6", "dog, wrong breed": "#eb6834", "cat": "#1baf7a"}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def photos():
    """Oxford test photos with their outline, true breed (ImageNet name, dogs only) and species."""
    for item in s3.list_photos("test"):
        if item["group"] == "food":
            continue
        name = Path(item["path"]).stem
        yield {"path": item["path"], "mask": PETS / "annotations" / "trimaps" / f"{name}.png",
               "species": item["group"], "breed": item["breed"]}


def analyse(batch, model, device, cats):
    xs, crops, masks = [], [], []
    for it in batch:
        x, crop = model_view(Image.open(it["path"]), EFFNET_B2.weights)
        xs.append(x)
        crops.append(crop)
        masks.append(mask_view(Image.open(it["mask"]), EFFNET_B2.weights))
    A = feature_maps(model, torch.cat(xs).to(device))
    A_fixed = fix_corner(A)                      # the same feature maps with the top-left artifact repaired

    def answers(feats):
        with torch.inference_mode():
            probs = torch.softmax(model.classifier(torch.flatten(model.avgpool(feats), 1)), dim=1).cpu()
        dog_part = probs[:, DOG_CLASSES.start:DOG_CLASSES.stop]
        return torch.where(torch.tensor([it["species"] == "dog" for it in batch]),
                           DOG_CLASSES.start + dog_part.argmax(1), probs.argmax(1))

    answer, answer_fixed = answers(A), answers(A_fixed)
    cams_raw = class_maps(model, A, answer.to(device)).cpu()            # what the app answered, raw map
    cams = class_maps(model, A_fixed, answer.to(device)).cpu()          # same answer, corner repaired
    rows = []
    for it, crop, mask, cam, cam_raw, a, a_fixed in zip(batch, crops, masks, cams, cams_raw, answer.tolist(),
                                                        answer_fixed.tolist()):
        if not mask.any():
            continue                             # a few Oxford outlines are empty (annotation errors): skip them
        heat = to_heat(cam[0], crop.size[0])
        heat_raw = to_heat(cam_raw[0], crop.size[0])
        said = cats[a]
        group = "cat" if it["species"] == "cat" else ("dog, right breed" if said == it["breed"] else "dog, wrong breed")
        if it["species"] == "dog" and it["breed"] is None:
            group = "dog, breed unknown to ImageNet"
        rows.append({"path": str(Path(it["path"]).relative_to(ROOT)), "group": group, "said": said,
                     "truth": it["breed"] or Path(it["path"]).stem.rsplit("_", 1)[0],
                     "heat_on_pet": heat_inside(heat, mask), "pet_area": float(mask.mean()),
                     "pointing_game": hottest_on_pet(heat, mask),
                     "raw_heat_on_pet": heat_inside(heat_raw, mask), "raw_pointing_game": hottest_on_pet(heat_raw, mask),
                     "said_without_artifact": cats[a_fixed], "_crop": crop, "_heat": heat, "_heat_raw": heat_raw})
    return rows


def plot(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    bins = np.linspace(0, 1, 26)
    for group, color in COLORS.items():
        vals = [r["heat_on_pet"] for r in rows if r["group"] == group and not np.isnan(r["heat_on_pet"])]
        ax.hist(vals, bins=bins, weights=np.full(len(vals), 100 / len(vals)), histtype="step", linewidth=2,
                color=color, label=f"{group} ({len(vals)} photos)")
    area = np.mean([r["pet_area"] for r in rows])
    ax.axvline(area, color=INK, linestyle="--", linewidth=1)
    ax.text(area - 0.01, 0.97, f"pet's average share of the picture: {area:.0%}\n(a heatmap spread evenly lands here)",
            transform=ax.get_xaxis_transform(), color=INK, fontsize=8.5, ha="right", va="top")
    ax.set_xlabel("share of the heat that falls on the pet", color=INK_2)
    ax.set_ylabel("% of photos in the group", color=INK_2)
    ax.set_title("Where does the model look? Oxford test photos", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.78))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def save_example(row, name):
    """The crop the model sees next to its heatmap, side by side, as a small JPEG."""
    crop, heat = row["_crop"], row["_heat"]
    pair = Image.new("RGB", (crop.size[0] * 2 + 8, crop.size[1]), (252, 252, 251))
    pair.paste(crop, (0, 0))
    pair.paste(overlay(crop, heat), (crop.size[0] + 8, 0))
    path = OUT / f"{name}.jpg"
    pair.save(path, quality=85)
    return str(path.relative_to(ROOT))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = EFFNET_B2.model.to(device)
    items = list(photos())
    print(f"{len(items)} Oxford test photos with outlines. Device: {device}")
    rows = []
    for start in range(0, len(items), 32):
        rows += analyse(items[start:start + 32], model, device, EFFNET_B2.categories)
        if start % 640 == 0:
            print(f"  {start + 32:>5} / {len(items)}", flush=True)
    EFFNET_B2.model.to("cpu")

    groups = {}
    for group in ["dog, right breed", "dog, wrong breed", "dog, breed unknown to ImageNet", "cat"]:
        g = [r for r in rows if r["group"] == group and not np.isnan(r["heat_on_pet"])]
        groups[group] = {"photos": len(g), "heat_on_pet": float(np.mean([r["heat_on_pet"] for r in g])),
                         "pet_area": float(np.mean([r["pet_area"] for r in g])),
                         "pointing_game": float(np.mean([r["pointing_game"] for r in g])),
                         "mostly_off_pet": float(np.mean([r["heat_on_pet"] < 0.5 for r in g])),
                         "raw_heat_on_pet": float(np.mean([r["raw_heat_on_pet"] for r in g])),
                         "raw_pointing_game": float(np.mean([r["raw_pointing_game"] for r in g])),
                         "answer_changes_without_artifact": float(np.mean([r["said"] != r["said_without_artifact"] for r in g]))}
        s = groups[group]
        print(f"  {group:32s} {s['photos']:5d} photos: heat on pet {s['heat_on_pet']:.0%} (pet area {s['pet_area']:.0%}), "
              f"hottest spot on pet {s['pointing_game']:.0%}, mostly off the pet {s['mostly_off_pet']:.0%} | raw: "
              f"heat on pet {s['raw_heat_on_pet']:.0%}, hottest on pet {s['raw_pointing_game']:.0%} | "
              f"answer changes without artifact {s['answer_changes_without_artifact']:.1%}")
    dogs = [r for r in rows if r["group"] in ("dog, right breed", "dog, wrong breed")]
    acc_raw = np.mean([r["group"] == "dog, right breed" for r in dogs])
    acc_fixed = np.mean([r["said_without_artifact"] == r["truth"] for r in dogs])
    print(f"  breed accuracy on the 21 breeds: {acc_raw:.1%} as is, {acc_fixed:.1%} with the corner repaired")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()
    dogs_ok = sorted([r for r in rows if r["group"] == "dog, right breed"], key=lambda r: r["heat_on_pet"])
    dogs_wrong = sorted([r for r in rows if r["group"] == "dog, wrong breed"], key=lambda r: r["heat_on_pet"])
    rng = np.random.default_rng(0)
    picks = ([("typical", r) for r in rng.choice(dogs_ok[len(dogs_ok) // 4:], 4, replace=False)]
             + [("off the dog, but right", r) for r in dogs_ok[:3]]
             + [("off the dog, and wrong", r) for r in dogs_wrong[:3]])
    examples = []
    for i, (kind, r) in enumerate(picks):
        examples.append({"kind": kind, "photo": r["path"], "said": r["said"], "truth": r["truth"],
                         "heat_on_pet": r["heat_on_pet"], "image": save_example(r, f"example_{i}")})
    # The artifact, shown once: raw map next to the repaired one, for the photo where it's most visible
    worst = max(dogs_ok, key=lambda r: r["heat_on_pet"] - r["raw_heat_on_pet"])
    crop = worst["_crop"]
    trio = Image.new("RGB", (crop.size[0] * 3 + 16, crop.size[1]), (252, 252, 251))
    for i, img in enumerate([crop, overlay(crop, worst["_heat_raw"]), overlay(crop, worst["_heat"])]):
        trio.paste(img, (i * (crop.size[0] + 8), 0))
    trio.save(OUT / "corner_artifact.jpg", quality=85)
    plot(rows, ROOT / "docs" / "stage7_heat_on_pet.png")
    report = {"model": EFFNET_B2.name, "method": "CAM on the last backbone block (equals Grad-CAM for this model), top-left cell repaired",
              "groups": groups, "examples": examples, "corner_example": {"photo": worst["path"], "image": "docs/stage7/corner_artifact.jpg"},
              "breed_accuracy": {"as_is": float(acc_raw), "corner_repaired": float(acc_fixed), "dogs": len(dogs)},
              "photos": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
    (ROOT / "reports" / "stage7_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("Saved reports/stage7_report.json, docs/stage7_heat_on_pet.png and docs/stage7/")


if __name__ == "__main__":
    main()
