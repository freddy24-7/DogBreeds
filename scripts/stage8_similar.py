"""
Stage 8: dogs that look like yours. A similarity search over 12,000 Stanford Dogs photos.

Run from the project folder (under a minute; it reuses stage 5's cached embeddings):
    uv run python scripts/stage8_similar.py

The steps (README_stage8.md explains each one):
  1. Choose how to compare embeddings, on validation photos only: gallery = 90% of Stanford
     train, queries = the other 10%. Rule set in advance: the smallest index whose
     precision@5 is within 0.5 points of the best.
  2. Build the final index on all 12,000 training photos and save it (models/stage8_index.npz).
  3. Look for copies: test photos whose nearest training photo is (almost) the same picture.
  4. Test the index once: Stanford test photos (8,580), the same without photos that have a
     same-shoot photo in the gallery, and Oxford dogs (a different photo collection).
  5. Can similarity alone tell a dog from a cat or food? Threshold chosen on stage 3's
     "choose" photos, checked on its test photos.
Writes reports/stage8_report.json, docs/stage8_*.png and example pictures in docs/stage8/.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import stage3_thresholds as s3  # noqa: E402
from stage5_train_head import imagenet_breed_probs_from_features, style  # noqa: E402
from dogbreeds.data import DATA, stanford_breeds, stanford_dogs  # noqa: E402
from dogbreeds.finetune import stratified_split  # noqa: E402
from dogbreeds.predictors import EFFNET_B2  # noqa: E402
from dogbreeds.similar import Index, same_breed_share, vote  # noqa: E402

INDEX_FILE = ROOT / "models" / "stage8_index.npz"
OUT = ROOT / "docs" / "stage8"
K = 10
VARIANTS = [  # how to compare embeddings; tried in step 1
    dict(key="cosine", label="cosine similarity, raw embeddings", centre=False, dims=None),
    dict(key="centred", label="centred, then cosine (all 1,408 numbers)", centre=True, dims=None),
    dict(key="pca256", label="centred, PCA to 256 numbers, cosine", centre=True, dims=256),
    dict(key="pca128", label="centred, PCA to 128 numbers, cosine", centre=True, dims=128),
    dict(key="pca64", label="centred, PCA to 64 numbers, cosine", centre=True, dims=64),
]
MAX_LOSS = 0.005          # step 1's rule: a smaller index may lose at most 0.5 points of precision@5
SAME_SHOOT = 0.88         # step 4: from this similarity on, the pairs we looked at were the same photo shoot
COPY = 0.99               # step 4: the same picture (resized or re-saved at most)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def load(name):
    z = np.load(DATA / f"stage5_features_b2_{name}.npz")
    return z["features"], z["labels"]


def scores(index, feats, labels, skip_above=None):
    sims, pos = index.search(feats, k=K, skip_above=skip_above)
    nb = index.labels[pos]
    return {"precision_at_1": float(same_breed_share(nb[:, :1], labels).mean()),
            "precision_at_5": float(same_breed_share(nb[:, :5], labels).mean()),
            "precision_at_10": float(same_breed_share(nb, labels).mean()),
            "knn_accuracy": float((vote(sims, nb, 120).argmax(1) == labels).mean())}


def grid(rows, path, size=150, gap=6):
    """Rows of (image path, caption) as one picture: first image = the query."""
    from PIL import ImageDraw
    n = max(len(r) for r in rows)
    canvas = Image.new("RGB", (n * (size + gap), len(rows) * (size + 34)), (252, 252, 251))
    draw = ImageDraw.Draw(canvas)
    for y, row in enumerate(rows):
        for x, (p, caption) in enumerate(row):
            im = Image.open(p).convert("RGB")
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                          (im.width + side) // 2, (im.height + side) // 2)).resize((size, size))
            canvas.paste(im, (x * (size + gap), y * (size + 34)))
            for i, line in enumerate(caption.split("\n")[:2]):
                draw.text((x * (size + gap) + 2, y * (size + 34) + size + 2 + 13 * i), line[:26],
                          fill=(11, 11, 11) if i == 0 else (82, 81, 78))
    canvas.save(path, quality=85)
    return str(path.relative_to(ROOT))


def plot_groups(top1, threshold, path):
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    style(ax)
    bins = np.linspace(0, 1, 41)
    for (group, vals), color in zip(top1.items(), SERIES):
        ax.hist(vals, bins=bins, weights=np.full(len(vals), 100 / len(vals)), histtype="step", linewidth=2,
                color=color, label=f"{group} ({len(vals)} photos)")
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1)
    ax.text(threshold - 0.01, 0.97, f"threshold {threshold:.2f}\nbelow: \"nothing like it\nin the gallery\"",
            transform=ax.get_xaxis_transform(), color=INK, fontsize=8.5, ha="right", va="top")
    ax.set_xlabel("similarity of the closest gallery photo", color=INK_2)
    ax.set_ylabel("% of photos in the group", color=INK_2)
    ax.set_title("How close is the nearest Stanford dog? Stage 3 test photos", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_copies(top1, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    style(ax)
    t = np.linspace(0.8, 1.0, 201)
    ax.plot(t, [(top1 >= x).sum() for x in t], color=SERIES[0], linewidth=2)
    for x, text in [(SAME_SHOOT, "same photo shoot\nfrom here on"), (COPY, "the same\npicture")]:
        n = int((top1 >= x).sum())
        ax.axvline(x, color=INK, linestyle="--", linewidth=1)
        ax.text(x - 0.004, n + 40, f"{text}\n{n} test photos", color=INK, fontsize=8.5, ha="right", va="bottom")
    ax.set_xlabel("similarity threshold", color=INK_2)
    ax.set_ylabel("test photos with a training\nphoto at least this similar", color=INK_2)
    ax.set_title("Stanford Dogs: test photos that are (nearly) in the training set", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    breeds = stanford_breeds(EFFNET_B2.categories)
    names = [b.name for b in breeds]
    train_ds, test_ds = stanford_dogs("train", breeds), stanford_dogs("test", breeds)
    train_paths = [str(p.relative_to(ROOT)) for p, _ in train_ds.items]
    test_paths = [p for p, _ in test_ds.items]
    X, y = load("stanford_train")
    Xt, yt = load("stanford_test")

    # ---- step 1: choose the variant on validation photos
    print("Step 1: choosing how to compare, on validation photos")
    tr, va = stratified_split(y, val_share=0.1, seed=0)
    variants = []
    for v in VARIANTS:
        index = Index.build(X[tr], y[tr], [train_paths[i] for i in tr], dims=v["dims"], centre=v["centre"])
        s = scores(index, X[va], y[va])
        variants.append({**v, **s, "numbers_per_photo": index.dims})
        print(f"  {v['label']:45s} precision@5 {s['precision_at_5']:.1%}  k-NN accuracy {s['knn_accuracy']:.1%}")
    best = max(v["precision_at_5"] for v in variants)
    ok = [v for v in variants if v["precision_at_5"] >= best - MAX_LOSS]
    chosen = min(ok, key=lambda v: (v["numbers_per_photo"], -v["precision_at_5"]))
    print(f"  chosen: {chosen['label']}")

    # ---- step 2: the final index on all training photos
    index = Index.build(X, y, train_paths, dims=chosen["dims"], centre=chosen["centre"])
    INDEX_FILE.parent.mkdir(exist_ok=True)
    index.save(INDEX_FILE)
    start = time.perf_counter()
    for i in range(100):
        index.search(Xt[i:i + 1], k=8)
    ms = (time.perf_counter() - start) * 10
    print(f"Step 2: saved {INDEX_FILE.relative_to(ROOT)} ({INDEX_FILE.stat().st_size / 1e6:.1f} MB), "
          f"{ms:.2f} ms per search")

    # ---- step 3: copies between test and training photos. Checked with ALL 1,408
    # numbers (centred), the space where we looked at pairs to set the thresholds. PCA'd similarities run
    # higher (fewer numbers left to disagree on), so the search index can't be used for this.
    check = Index.build(X, y, train_paths)
    sims, pos = check.search(Xt, k=1)
    top1_test, pos1 = sims[:, 0], pos[:, 0]
    copies = top1_test >= COPY
    same_shoot = top1_test >= SAME_SHOOT
    conflicts = copies & (y[pos1] != yt)
    tr_sims, tr_pos = check.search(X, k=2)                      # within the training set: [0] is the photo itself
    train_copies = tr_sims[:, 1] >= COPY
    dup = {"test_photos": len(yt), "same_shoot": int(same_shoot.sum()), "copies": int(copies.sum()),
           "copies_with_a_different_breed": int(conflicts.sum()),
           "train_photos_with_a_copy_in_train": int(train_copies.sum()),
           "of_which_different_breed": int((train_copies & (y[tr_pos[:, 1]] != y)).sum())}
    print(f"Step 3: {dup}")

    # ---- step 4: test; and again on only the test photos with no same-shoot photo in the gallery
    test = {"stanford": scores(index, Xt, yt), "stanford_no_same_shoot": scores(index, Xt[~same_shoot], yt[~same_shoot])}
    test["stanford_no_same_shoot"]["photos"] = int((~same_shoot).sum())
    head = imagenet_breed_probs_from_features(Xt, breeds).argmax(1)
    test["stanford"]["imagenet_head_accuracy"] = float((head == yt).mean())
    Xc, yc = load("stage3_choose")
    Xs, ys = load("stage3_test")
    dogs = ys >= 0
    test["oxford_dogs"] = scores(index, Xs[dogs], ys[dogs])
    test["oxford_dogs"]["imagenet_head_accuracy"] = float(
        (imagenet_breed_probs_from_features(Xs[dogs], breeds).argmax(1) == ys[dogs]).mean())
    for k, s in test.items():
        print(f"Step 4: {k:24s} " + "  ".join(f"{m} {v:.1%}" if v <= 1 else f"{m} {v}" for m, v in s.items()))

    # ---- step 5: does a photo have anything like it in the gallery?
    groups_c = [it["group"] for it in s3.list_photos("trainval")]
    groups_s = [it["group"] for it in s3.list_photos("test")]
    top1_c = index.search(Xc, k=1)[0][:, 0]
    top1_s = index.search(Xs, k=1)[0][:, 0]
    dog_c = np.array([g == "dog" for g in groups_c])
    threshold = float(np.quantile(top1_c[dog_c], 0.01))       # keeps 99% of dogs (choose photos)
    by_group = {g: top1_s[np.array(groups_s) == g] for g in ["dog", "cat", "food"]}
    unusual = {g: {"photos": int(len(v)), "median_similarity": float(np.median(v)),
                   "flagged": float((v < threshold).mean())} for g, v in by_group.items()}
    print(f"Step 5: threshold {threshold:.3f} -> " + ", ".join(f"{g} flagged {u['flagged']:.1%}" for g, u in unusual.items()))

    # ---- pictures
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()
    plot_groups({f"{g}s" if g != "food" else g: v for g, v in by_group.items()}, threshold,
                ROOT / "docs" / "stage8_similarity_by_group.png")
    plot_copies(top1_test, ROOT / "docs" / "stage8_copies.png")
    rng = np.random.default_rng(1)
    oxford_paths = [it["path"] for it in s3.list_photos("test")]
    dog_rows = np.flatnonzero(dogs)
    queries = rng.choice(dog_rows, 5, replace=False)
    qs, qp = index.search(Xs[queries], k=6)
    rows = [[(oxford_paths[q], f"your photo:\n{names[ys[q]]}")]
            + [(ROOT / index.paths[p], f"{names[index.labels[p]]}\nsimilarity {s:.2f}") for s, p in zip(srow, prow)]
            for q, srow, prow in zip(queries, qs, qp)]
    neighbours_image = grid(rows, OUT / "neighbours.jpg")
    conflict_rows = [[(test_paths[q], f"test: {names[yt[q]]}"),
                      (ROOT / check.paths[pos1[q]], f"train: {names[y[pos1[q]]]}\nsimilarity {top1_test[q]:.3f}")]
                     for q in np.flatnonzero(conflicts)[:8]]
    pairs = [conflict_rows[i] + (conflict_rows[i + 1] if i + 1 < len(conflict_rows) else [])
             for i in range(0, len(conflict_rows), 2)]
    conflicts_image = grid(pairs, OUT / "copies_with_two_breeds.jpg")
    shoot = np.flatnonzero(same_shoot & ~copies)
    shoot = shoot[np.argsort(top1_test[shoot])][:: max(1, len(shoot) // 4)][:4]
    shoot_rows = [[(test_paths[q], f"test: {names[yt[q]]}"),
                   (ROOT / check.paths[pos1[q]], f"train: {names[y[pos1[q]]]}\nsimilarity {top1_test[q]:.3f}")]
                  for q in shoot]
    shoot_image = grid([shoot_rows[0] + shoot_rows[1], shoot_rows[2] + shoot_rows[3]], OUT / "same_shoot.jpg")
    examples = [str(Path(oxford_paths[q]).relative_to(ROOT)) for q in queries]
    cat_example = next(p for p, g in zip(oxford_paths, groups_s) if g == "cat")
    examples.append(str(Path(cat_example).relative_to(ROOT)))

    report = {"model": EFFNET_B2.name, "gallery": "Stanford Dogs train (12,000 photos)", "k": K,
              "variants": variants, "chosen": chosen["key"], "rule": f"smallest index within {MAX_LOSS:.1%} of the best precision@5",
              "index": {"file": str(INDEX_FILE.relative_to(ROOT)), "megabytes": INDEX_FILE.stat().st_size / 1e6,
                        "numbers_per_photo": index.dims, "ms_per_search": ms},
              "test": test, "copies": dup, "same_shoot_threshold": SAME_SHOOT, "copy_threshold": COPY,
              "unusual": {"threshold": threshold, "groups": unusual},
              "images": {"neighbours": neighbours_image, "copies_with_two_breeds": conflicts_image,
                         "same_shoot": shoot_image},
              "examples": examples}
    (ROOT / "reports" / "stage8_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("Saved reports/stage8_report.json, docs/stage8_*.png and docs/stage8/")


if __name__ == "__main__":
    main()
