"""
Stage 3: measure where to put the "no dog" and "unsure" thresholds.

Run from the project folder (takes a few minutes):
    uv run python scripts/stage3_thresholds.py

What it does:
  1. Runs both ImageNet models on photos of dogs and cats (Oxford-IIIT Pets)
     and food (pizza/steak/sushi), and records each photo's dog share and
     breed confidence.
  2. Chooses the thresholds on the "trainval" photos only.
  3. Reports how well they work on the separate "test" photos, which played
     no part in choosing them (so the result isn't flattered).
  4. Saves config/stage3_thresholds.json (read by the app) and one plot per
     model in docs/.

Needs data/oxford-iiit-pet (downloaded by torchvision) and
data/pizza_steak_sushi (copied from the course project).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dogbreeds.predictors import DOG_CLASSES, EFFNET_B0, EFFNET_B2  # noqa: E402

DATA = ROOT / "data"
PETS = DATA / "oxford-iiit-pet"
FOOD = DATA / "pizza_steak_sushi"

# The rules we choose thresholds by. Change these, rerun, and the app follows.
KEEP_DOGS = 0.98        # the dog threshold may wrongly reject at most 2% of real dogs
TARGET_ACCURACY = 0.95  # a named ("green") breed must be right at least 95% of the time

# Oxford Pets dog breeds -> the same breed's ImageNet name. Four Oxford breeds have no
# ImageNet equivalent (American bulldog, American pit bull terrier, Havanese, Shiba Inu);
# their photos still count as dogs, but not for breed accuracy.
OXFORD_TO_IMAGENET = {
    "basset_hound": "basset", "beagle": "beagle", "boxer": "boxer", "chihuahua": "Chihuahua",
    "english_cocker_spaniel": "cocker spaniel", "english_setter": "English setter",
    "german_shorthaired": "German short-haired pointer", "great_pyrenees": "Great Pyrenees",
    "japanese_chin": "Japanese spaniel", "keeshond": "keeshond", "leonberger": "Leonberg",
    "miniature_pinscher": "miniature pinscher", "newfoundland": "Newfoundland",
    "pomeranian": "Pomeranian", "pug": "pug", "saint_bernard": "Saint Bernard", "samoyed": "Samoyed",
    "scottish_terrier": "Scotch terrier", "staffordshire_bull_terrier": "Staffordshire bullterrier",
    "wheaten_terrier": "soft-coated wheaten terrier", "yorkshire_terrier": "Yorkshire terrier",
}

# Colours: the first three slots of the validated default categorical palette (dataviz skill)
COLORS = {"dog": "#2a78d6", "cat": "#eb6834", "food": "#1baf7a"}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def list_photos(split: str) -> list[dict]:
    """All photos of one split: path, group (dog / cat / food) and true breed (dogs only)."""
    items = []
    # Oxford Pets annotation lines look like: "beagle_101 13 2 3" = name, class, species (1 cat, 2 dog), breed
    for line in (PETS / "annotations" / f"{split}.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, species, _ = line.split()
        breed = name.rsplit("_", 1)[0]
        group = "dog" if species == "2" else "cat"
        items.append(dict(path=PETS / "images" / f"{name}.jpg", group=group,
                          breed=OXFORD_TO_IMAGENET.get(breed) if group == "dog" else None))
    food_split = "train" if split == "trainval" else "test"
    for path in sorted((FOOD / food_split).glob("*/*.jpg")):
        items.append(dict(path=path, group="food", breed=None))
    return items


class Photos(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(Image.open(self.items[i]["path"]).convert("RGB"))


def score(predictor, items, device) -> list[dict]:
    """Runs the model on all photos in batches (much faster than one by one)."""
    model = predictor.model.to(device)
    loader = DataLoader(Photos(items, predictor.weights.transforms()), batch_size=64, num_workers=0)
    dog_names = [predictor.categories[i] for i in DOG_CLASSES]
    rows = []
    with torch.inference_mode():
        for x in loader:
            probs = torch.softmax(model(x.to(device)), dim=1).cpu()
            dog_probs = probs[:, DOG_CLASSES.start:DOG_CLASSES.stop]
            dog_share = dog_probs.sum(dim=1)
            top3 = dog_probs.topk(3, dim=1)
            for share, vals, idxs in zip(dog_share.tolist(), top3.values.tolist(), top3.indices.tolist()):
                rows.append(dict(dog_share=share, confidence=vals[0] / max(share, 1e-12),
                                 top3=[dog_names[i] for i in idxs]))
    predictor.model.to("cpu")  # the app predicts on the CPU
    return [{**item, **row} for item, row in zip(items, rows)]


def choose_thresholds(rows) -> tuple[float, float]:
    dog_shares = np.array([r["dog_share"] for r in rows if r["group"] == "dog"])
    # The 2nd percentile: 98% of dogs have a higher dog share. Rounded down to 2 decimals.
    dog_t = float(np.floor(np.quantile(dog_shares, 1 - KEEP_DOGS) * 100) / 100)

    named = [r for r in rows if r["breed"] and r["dog_share"] >= dog_t]
    breed_t = 0.99
    for t in np.round(np.arange(0, 1, 0.01), 2):           # lowest threshold that reaches the target
        covered = [r for r in named if r["confidence"] >= t]
        if covered and np.mean([r["top3"][0] == r["breed"] for r in covered]) >= TARGET_ACCURACY:
            breed_t = float(t)
            break
    return dog_t, breed_t


def report(rows, dog_t, breed_t) -> dict:
    def share(rs, cond):
        return round(float(np.mean([cond(r) for r in rs])), 4) if rs else None

    by = {g: [r for r in rows if r["group"] == g] for g in COLORS}
    mapped = [r for r in by["dog"] if r["breed"]]
    accepted = [r for r in mapped if r["dog_share"] >= dog_t]
    green = [r for r in accepted if r["confidence"] >= breed_t]
    yellow = [r for r in accepted if r["confidence"] < breed_t]
    return {
        "photos": {g: len(rs) for g, rs in by.items()},
        "dogs_accepted": share(by["dog"], lambda r: r["dog_share"] >= dog_t),
        "cats_rejected": share(by["cat"], lambda r: r["dog_share"] < dog_t),
        "food_rejected": share(by["food"], lambda r: r["dog_share"] < dog_t),
        "breed_top1_accuracy_all_dogs": share(mapped, lambda r: r["top3"][0] == r["breed"]),
        "breed_top3_accuracy_all_dogs": share(mapped, lambda r: r["breed"] in r["top3"]),
        "dogs_named_green": share(accepted, lambda r: r["confidence"] >= breed_t),
        "green_accuracy": share(green, lambda r: r["top3"][0] == r["breed"]),
        "yellow_true_breed_in_top3": share(yellow, lambda r: r["breed"] in r["top3"]),
    }


def plot(rows, dog_t, breed_t, title, path):
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": GRID})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # Left: where the dog share of dogs, cats and food falls
    bins = np.linspace(0, 1, 41)
    for group, color in COLORS.items():
        vals = [r["dog_share"] for r in rows if r["group"] == group]
        weights = np.full(len(vals), 100 / len(vals))  # percent of that group, so groups compare fairly
        ax1.hist(vals, bins=bins, weights=weights, histtype="step", linewidth=2, color=color)
    handles = [Line2D([], [], color=c, linewidth=2) for c in COLORS.values()]
    labels = [f"{g} ({sum(r['group'] == g for r in rows)} photos)" for g in COLORS]
    ax1.set_yscale("log")
    ax1.axvline(dog_t, color=INK, linestyle="--", linewidth=1)
    ax1.text(dog_t + 0.01, 0.5, f"threshold {dog_t:.2f}", transform=ax1.get_xaxis_transform(), color=INK)
    ax1.set_xlabel("dog share (probability on the 118 dog breeds)")
    ax1.set_ylabel("% of photos in the group (log scale)")
    ax1.set_title("Is there a dog? Dog share per photo", loc="left", color=INK)
    ax1.legend(handles, labels, frameon=False, loc="upper center")

    # Right: the effect of the breed threshold on dogs with a known ImageNet breed
    named = [r for r in rows if r["breed"] and r["dog_share"] >= dog_t]
    ts = np.round(np.arange(0, 1, 0.01), 2)
    acc, cov = [], []
    for t in ts:
        covered = [r for r in named if r["confidence"] >= t]
        cov.append(100 * len(covered) / len(named))
        acc.append(100 * np.mean([r["top3"][0] == r["breed"] for r in covered]) if covered else np.nan)
    ax2.plot(ts, acc, color=COLORS["dog"], linewidth=2, label="accuracy of the named breed")
    ax2.plot(ts, cov, color=INK_2, linewidth=2, linestyle=":", label="% of dogs that get a breed name")
    ax2.axvline(breed_t, color=INK, linestyle="--", linewidth=1)
    ax2.text(breed_t + 0.01, 0.06, f"threshold {breed_t:.2f}", transform=ax2.get_xaxis_transform(), color=INK)
    ax2.set_ylim(0, 101)
    ax2.set_xlabel("breed confidence threshold")
    ax2.set_ylabel("%")
    ax2.set_title("Which breed? Accuracy vs how often we answer", loc="left", color=INK)
    ax2.legend(frameon=False, loc="lower left")

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    choose_items, test_items = list_photos("trainval"), list_photos("test")
    print(f"{len(choose_items)} photos to choose thresholds, {len(test_items)} to test them. Device: {device}")

    out = {"how": {"chosen_on": "Oxford-IIIT Pets trainval + pizza_steak_sushi train",
                   "tested_on": "Oxford-IIIT Pets test + pizza_steak_sushi test",
                   "keep_dogs": KEEP_DOGS, "target_accuracy": TARGET_ACCURACY},
           "models": {}}
    (ROOT / "docs").mkdir(exist_ok=True)
    for key, predictor in [("b0", EFFNET_B0), ("b2", EFFNET_B2)]:
        print(f"\n{predictor.name}: scoring...")
        cache = DATA / f"stage3_scores_{key}.json"   # the model's output per photo (not in git)
        if cache.exists():
            choose_rows, test_rows = json.loads(cache.read_text())
            print("  (using saved scores, delete data/stage3_scores_*.json to rerun the model)")
        else:
            choose_rows = score(predictor, choose_items, device)
            test_rows = score(predictor, test_items, device)
            cache.write_text(json.dumps([choose_rows, test_rows], default=str))
        dog_t, breed_t = choose_thresholds(choose_rows)
        result = report(test_rows, dog_t, breed_t)
        plot_path = ROOT / "docs" / f"stage3_thresholds_{key}.png"
        plot(test_rows, dog_t, breed_t, f"{predictor.name}, test photos", plot_path)
        out["models"][predictor.name] = {"dog_threshold": dog_t, "breed_threshold": breed_t,
                                         "plot": str(plot_path.relative_to(ROOT)), "test": result}
        print(f"  dog threshold {dog_t}, breed threshold {breed_t}")
        print("  " + json.dumps(result, indent=2).replace("\n", "\n  "))

    (ROOT / "config").mkdir(exist_ok=True)
    (ROOT / "config" / "stage3_thresholds.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nSaved config/stage3_thresholds.json and the plots in docs/")


if __name__ == "__main__":
    main()
