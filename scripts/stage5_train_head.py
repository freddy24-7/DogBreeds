"""
Stage 5: train our own head on top of the frozen EfficientNet-B2 backbone.

Run from the project folder:
    uv run python scripts/stage5_train_head.py                  # everything
    uv run python scripts/stage5_train_head.py --features-only  # only step 1

The steps (README_stage5.md explains each one):
  1. Cache features: run the frozen backbone once on every photo and save its
     1,408 numbers per photo in data/ (about 30 minutes the first time).
  2. Check the cache: ImageNet's own head on the cached features must give the
     same predictions as stage 4.
  3. Split Stanford Dogs train into train (90%) and validation (10%).
  4. Train the head with a few settings; choose the best on the validation photos.
  5. Test the chosen head once on the stage 4 test photos and the Oxford check.
  6. Measure its stage 3 thresholds (it has its own confidence, so its own thresholds).
  7. Save models/stage5_head_b2.pth, reports/stage5_report.json,
     config/stage5_thresholds.json and plots in docs/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import stage3_thresholds as s3  # noqa: E402  (reuse stage 3's photo lists and threshold rules)
from dogbreeds.data import DATA, PhotoDataset, stanford_breeds, stanford_dogs  # noqa: E402
from dogbreeds.evaluate import model_report, topk_accuracy  # noqa: E402
from dogbreeds.finetune import (BreedHead, extract_features, head_from_imagenet, predict_probs,  # noqa: E402
                                save_head, stratified_split, train_head)
from dogbreeds.predictors import DOG_CLASSES, EFFNET_B2, FINETUNED_NAME, HEAD_FILE  # noqa: E402

EPOCHS = 30
SEED = 0
# The settings we try. Each run takes seconds, because it only trains the head on saved features.
EXPERIMENTS = [
    dict(key="random_lr0.001", start="random", lr=1e-3, label="new head, random start, learning rate 0.001"),
    dict(key="random_lr0.01", start="random", lr=1e-2, label="new head, random start, learning rate 0.01"),
    dict(key="imagenet_lr0.0001", start="imagenet", lr=1e-4, label="start from ImageNet's head, learning rate 0.0001"),
    dict(key="imagenet_lr0.001", start="imagenet", lr=1e-3, label="start from ImageNet's head, learning rate 0.001"),
]

# Validated default categorical palette (dataviz skill), slots 1-4 in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
UP, DOWN = "#2a78d6", "#eb6834"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


# ---------------------------------------------------------------- step 1: cache features

def photo_sets(breeds) -> dict[str, tuple[PhotoDataset, list[str] | None]]:
    """Every photo set we need features for, as (dataset, group per photo)."""
    name_to_index = {b.name: b.index for b in breeds}
    sets = {"stanford_train": (stanford_dogs("train", breeds), None),
            "stanford_test": (stanford_dogs("test", breeds), None)}
    # Stage 3's photos (Oxford dogs and cats, plus food). Label -1 = not one of our 120 breeds (a cat, food,
    # or one of the 4 Oxford dog breeds Stanford doesn't have).
    for split, name in [("trainval", "stage3_choose"), ("test", "stage3_test")]:
        items = s3.list_photos(split)
        ds = PhotoDataset([(it["path"], name_to_index.get(it["breed"], -1)) for it in items])
        sets[name] = (ds, [it["group"] for it in items])
    return sets


def cached_features(name: str, dataset: PhotoDataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / f"stage5_features_b2_{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["features"], z["labels"]
    print(f"  computing features for {name} ({len(dataset)} photos)...", flush=True)
    feats, labels = extract_features(EFFNET_B2, dataset, device)
    np.savez(path, features=feats, labels=labels)
    return feats, labels


# ---------------------------------------------------------------- helpers

def imagenet_breed_probs_from_features(feats, breeds) -> np.ndarray:
    """ImageNet's own head on cached features, as 120 breed probabilities (as in stage 4)."""
    with torch.inference_mode():
        logits = EFFNET_B2.model.classifier.eval()(torch.as_tensor(feats))
    probs = torch.softmax(logits, dim=1)[:, [b.imagenet_index for b in breeds]]
    return (probs / probs.sum(dim=1, keepdim=True)).numpy()


def dog_shares(feats) -> np.ndarray:
    """ImageNet's 'is it a dog?' answer (stage 3): total probability on the 118 dog classes."""
    with torch.inference_mode():
        probs = torch.softmax(EFFNET_B2.model.classifier.eval()(torch.as_tensor(feats)), dim=1)
    return probs[:, DOG_CLASSES.start:DOG_CLASSES.stop].sum(dim=1).numpy()


def report_set(probs, labels, breeds, mask=None) -> dict:
    if mask is not None:
        probs, labels = probs[mask], labels[mask]
    return model_report(probs, labels, breeds)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- plots

def plot_training(experiments, chosen_key, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        style(ax)
    for exp, color in zip(experiments, SERIES):
        h = exp["history"]
        ax1.plot([e["epoch"] for e in h], [100 * e["val_accuracy"] for e in h], color=color, linewidth=2,
                 label=exp["label"])
    ax1.set_xlabel("epoch (one pass over the 10,800 training photos)", color=INK_2)
    ax1.set_ylabel("validation accuracy (%)", color=INK_2)
    lo = min(100 * e["val_accuracy"] for exp in experiments for e in exp["history"][1:])
    ax1.set_ylim(max(lo - 2, 0), 100)
    ax1.set_title("Four settings, compared on validation photos", loc="left", color=INK)
    ax1.text(1.3, 99.2, "random start: 0.7% at epoch 0 (pure guessing)", color=INK_2, fontsize=8.5)
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5)

    chosen = next(e for e in experiments if e["key"] == chosen_key)
    h = chosen["history"]
    ax2.plot([e["epoch"] for e in h[1:]], [e["train_loss"] for e in h[1:]], color=SERIES[0], linewidth=2,
             label="training photos (dropout on)")
    ax2.plot([e["epoch"] for e in h], [e["val_loss"] for e in h], color=SERIES[1], linewidth=2,
             label="validation photos")
    ax2.axvline(chosen["best_epoch"], color=INK, linestyle="--", linewidth=1)
    ax2.text(chosen["best_epoch"] + 0.4, 0.92, f"kept: epoch {chosen['best_epoch']}",
             transform=ax2.get_xaxis_transform(), color=INK, fontsize=9)
    ax2.set_xlabel("epoch", color=INK_2)
    ax2.set_ylabel("loss (lower is better)", color=INK_2)
    ax2.set_title(f"Loss of the chosen setting ({'ImageNet' if chosen['start'] == 'imagenet' else 'random'} start, lr {chosen['lr']:g})", loc="left", color=INK)
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_breed_changes(before: dict, after: dict, path, n=8):
    diffs = sorted(((after[b] - before[b], b) for b in before), reverse=True)
    rows = [d for d in diffs[:n] if d[0] > 0] + [d for d in diffs[-n:] if d[0] < 0]
    rows = rows[::-1]   # biggest gain at the top
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2), facecolor=SURFACE)
    style(ax)
    ax.grid(False)
    vals = [100 * d for d, _ in rows]
    ax.barh([b for _, b in rows], vals, color=[UP if v > 0 else DOWN for v in vals], height=0.6)
    for i, ((d, b), v) in enumerate(zip(rows, vals)):
        ax.text(v + (0.4 if v > 0 else -0.4), i, f"{100 * before[b]:.0f}% → {100 * after[b]:.0f}%", va="center",
                ha="left" if v > 0 else "right", color=INK_2, fontsize=8.5)
    ax.axvline(0, color=INK, linewidth=1)
    lim = max(abs(v) for v in vals) * 1.6
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("change in top-1 accuracy (percentage points), Stanford Dogs test", color=INK_2)
    ax.set_title("Biggest changes per breed: ImageNet's head → ours", loc="left", color=INK)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    breeds = stanford_breeds(EFFNET_B2.categories)
    names = [b.name for b in breeds]

    print("Step 1: cache features (the frozen backbone runs once per photo)")
    sets = photo_sets(breeds)
    F = {name: cached_features(name, ds, device) for name, (ds, _) in sets.items()}
    if "--features-only" in sys.argv:
        return

    print("Step 2: check the cache against stage 4")
    test_x, test_y = F["stanford_test"]
    base_test = imagenet_breed_probs_from_features(test_x, breeds)
    stage4 = np.load(DATA / "stage4_preds_b2_stanford_test.npz")
    sanity = {"max_probability_difference": float(np.abs(base_test - stage4["probs"]).max()),
              "same_top1_share": float((base_test.argmax(1) == stage4["probs"].argmax(1)).mean()),
              "top1_from_cache": topk_accuracy(base_test, test_y, 1)}
    print(f"  {sanity}")
    assert sanity["same_top1_share"] > 0.995, "cached features don't reproduce stage 4"

    print("Step 3: split Stanford Dogs train into train and validation")
    x, y = F["stanford_train"]
    tr, va = stratified_split(y, val_share=0.1, seed=SEED)
    print(f"  {len(tr)} training photos, {len(va)} validation photos ({len(va) // 120} per breed)")

    print("Step 4: train the head with each setting")
    imagenet_linear = EFFNET_B2.model.classifier[1]
    experiments, states = [], {}
    for exp in EXPERIMENTS:
        torch.manual_seed(SEED)   # the same random start for every 'random' run
        head = (head_from_imagenet(imagenet_linear, [b.imagenet_index for b in breeds]) if exp["start"] == "imagenet"
                else BreedHead(x.shape[1], len(breeds)))
        state, history, best_epoch = train_head(head, x[tr], y[tr], x[va], y[va], epochs=EPOCHS, lr=exp["lr"],
                                                seed=SEED)
        best = history[best_epoch]
        experiments.append({**exp, "best_epoch": best_epoch, "val_accuracy": best["val_accuracy"],
                            "val_loss": best["val_loss"], "history": history})
        states[exp["key"]] = state
        print(f"  {exp['label']:52s} best epoch {best_epoch:2d}: val accuracy {best['val_accuracy']:.1%}, "
              f"val loss {best['val_loss']:.3f}  (epoch 0: {history[0]['val_accuracy']:.1%})")
    chosen = max(experiments, key=lambda e: (round(e["val_accuracy"], 4), -e["val_loss"]))
    print(f"  chosen on validation photos: {chosen['label']}")
    head = BreedHead(x.shape[1], len(breeds))
    head.load_state_dict(states[chosen["key"]])
    head.eval()

    print("Step 5: test once, on the same photos as stage 4")
    s3_test_x, s3_test_y = F["stage3_test"]
    ox = s3_test_y >= 0           # Oxford dogs of the 21 breeds Stanford also has (stage 4's Oxford check)
    oxford_breeds = np.unique(s3_test_y[ox])
    s21 = np.isin(test_y, oxford_breeds)
    test = {}
    for model_key, probs_fn in [("imagenet_head", lambda f: imagenet_breed_probs_from_features(f, breeds)),
                                ("our_head", lambda f: predict_probs(head, f))]:
        p_st, p_ox = probs_fn(test_x), probs_fn(s3_test_x)
        test[model_key] = {"stanford_test": report_set(p_st, test_y, breeds),
                           "stanford_test_21_breeds": {"photos": int(s21.sum()),
                                                       "top1": topk_accuracy(p_st[s21], test_y[s21], 1),
                                                       "top3": topk_accuracy(p_st[s21], test_y[s21], 3)},
                           "oxford_test": report_set(p_ox, s3_test_y, breeds, ox)}
        r = test[model_key]
        print(f"  {model_key:13s} Stanford top-1 {r['stanford_test']['top1']:.1%} (21 breeds "
              f"{r['stanford_test_21_breeds']['top1']:.1%}), Oxford top-1 {r['oxford_test']['top1']:.1%}, "
              f"ECE {r['stanford_test']['ece']:.3f} / {r['oxford_test']['ece']:.3f}")
    assert test["imagenet_head"]["oxford_test"]["photos"] == 2086

    print("Step 6: stage 3 thresholds for our head")
    thresholds = {}
    for split, set_name in [("trainval", "stage3_choose"), ("test", "stage3_test")]:
        fx, _ = F[set_name]
        items = s3.list_photos(split)
        shares = dog_shares(fx)
        probs = predict_probs(head, fx)
        top3 = np.argsort(-probs, axis=1)[:, :3]
        thresholds[split] = [{**it, "dog_share": float(s), "confidence": float(p[t[0]]),
                              "top3": [names[i] for i in t]} for it, s, p, t in zip(items, shares, probs, top3)]
    dog_t, breed_t = s3.choose_thresholds(thresholds["trainval"])
    th_result = s3.report(thresholds["test"], dog_t, breed_t)
    (ROOT / "docs").mkdir(exist_ok=True)
    s3.plot(thresholds["test"], dog_t, breed_t, f"{FINETUNED_NAME}, test photos", ROOT / "docs" / "stage5_thresholds.png")
    print(f"  dog threshold {dog_t}, breed threshold {breed_t}: named 🟢 {th_result['dogs_named_green']:.1%} "
          f"at {th_result['green_accuracy']:.1%} accuracy")
    (ROOT / "config" / "stage5_thresholds.json").write_text(json.dumps({
        "how": {"chosen_on": "Oxford-IIIT Pets trainval + pizza_steak_sushi train (same rules as stage 3)",
                "tested_on": "Oxford-IIIT Pets test + pizza_steak_sushi test",
                "keep_dogs": s3.KEEP_DOGS, "target_accuracy": s3.TARGET_ACCURACY},
        "models": {FINETUNED_NAME: {"dog_threshold": dog_t, "breed_threshold": breed_t,
                                    "plot": "docs/stage5_thresholds.png", "test": th_result}}}, indent=2) + "\n")

    # Example photos for the app: unseen Oxford test dogs where the two heads disagree
    p_base, p_ours = imagenet_breed_probs_from_features(s3_test_x, breeds), predict_probs(head, s3_test_x)
    paths = [str(p.relative_to(ROOT)) for p, _ in sets["stage3_test"][0].items]
    fixed = np.flatnonzero(ox & (p_base.argmax(1) != s3_test_y) & (p_ours.argmax(1) == s3_test_y))
    broken = np.flatnonzero(ox & (p_base.argmax(1) == s3_test_y) & (p_ours.argmax(1) != s3_test_y))
    rng = np.random.default_rng(SEED)
    examples = ([{"path": paths[i], "note": "fixed by our head"} for i in rng.choice(fixed, min(4, len(fixed)), replace=False)]
                + [{"path": paths[i], "note": "broken by our head"} for i in rng.choice(broken, min(2, len(broken)), replace=False)])
    print(f"  Oxford test: our head fixes {len(fixed)} photos and breaks {len(broken)}")

    print("Step 7: save")
    save_head(HEAD_FILE, head, names, {"base": "EfficientNet-B2 (ImageNet), frozen", "trained_on":
              f"Stanford Dogs train, {len(tr)} photos", "setting": chosen["label"], "epoch": chosen["best_epoch"]})
    plot_training(experiments, chosen["key"], ROOT / "docs" / "stage5_training.png")
    plot_breed_changes(test["imagenet_head"]["stanford_test"]["per_breed"], test["our_head"]["stanford_test"]["per_breed"],
                       ROOT / "docs" / "stage5_breed_changes.png")
    out = {"model": FINETUNED_NAME, "cache_check": sanity,
           "split": {"train": len(tr), "validation": len(va), "test": len(test_y)},
           "experiments": experiments, "chosen": chosen["key"], "test": test,
           "oxford_fixed": len(fixed), "oxford_broken": len(broken), "examples": examples,
           "thresholds": {"dog": dog_t, "breed": breed_t, "test": th_result}}
    (ROOT / "reports" / "stage5_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Saved models/stage5_head_b2.pth, reports/stage5_report.json, config/stage5_thresholds.json, docs/stage5_*.png")


if __name__ == "__main__":
    main()
