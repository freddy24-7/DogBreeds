"""
Stage 6: fine-tune the last blocks of the backbone, with data augmentation.

Run from the project folder (about an hour the first time, on an M1 with 8 GB):
    uv run python scripts/stage6_finetune.py              # train (or reuse the saved model) and report
    uv run python scripts/stage6_finetune.py --retrain    # train again from scratch

The steps (README_stage6.md explains each one):
  1. The same train / validation split as stage 5 (the same 10 photos per breed set aside).
  2. Build the model: EfficientNet-B2, blocks 0-6 frozen, blocks 7-8 trainable,
     and a head that starts as ImageNet's head (stage 5's best start).
  3. Train on real photos with data augmentation; keep the epoch with the lowest
     validation loss. The best weights are saved after every epoch.
  4. Test once on the stage 4 photos and the Oxford check, next to stage 4 and stage 5.
  5. Error analysis: what happened to the look-alike pairs from stages 4 and 5.
  6. Measure its own stage 3 thresholds, then apply the ship rule (decided before training).
  7. Save reports/stage6_report.json, config/stage6_thresholds.json and plots in docs/.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import stage3_thresholds as s3  # noqa: E402
import stage5_train_head as s5  # noqa: E402
from dogbreeds.data import DATA, PhotoDataset, stanford_breeds, stanford_dogs  # noqa: E402
from dogbreeds.evaluate import confusion_matrix, topk_accuracy  # noqa: E402
from dogbreeds.finetune import (PartlyFrozen, head_from_imagenet, load_head, predict_probs,  # noqa: E402
                                predict_photos, stratified_split, train_partly_frozen, train_transform)
from dogbreeds.predictors import EFFNET_B2, HEAD_FILE, STAGE6_FILE, STAGE6_NAME  # noqa: E402

FIRST_TRAINED_BLOCK = 7   # train blocks 7 and 8 (3.2M of the backbone's 7.7M weights), freeze 0-6
EPOCHS = 6
LR = 1e-4                 # 10x smaller than stage 5's head: pre-trained weights need small nudges
SEED = 0

# The ship rule, decided BEFORE training so the result can't bend it:
MIN_OXFORD_GAIN = 0.01    # at least +1 point top-1 on the unseen Oxford photos (more than the noise)
                          # ...and at least as many dogs get a 🟢 name (stage 3 rules) as with ImageNet's head

INK, INK_2, SURFACE, GRID = s5.INK, s5.INK_2, s5.SURFACE, s5.GRID
BLUE, ORANGE = s5.SERIES[0], s5.SERIES[1]


def build_model(breeds) -> PartlyFrozen:
    """A fresh EfficientNet-B2 (not the app's shared one), partly frozen, with an ImageNet-start head."""
    base = torchvision.models.efficientnet_b2(weights=EFFNET_B2.weights)
    head = head_from_imagenet(base.classifier[1], [b.imagenet_index for b in breeds])
    return PartlyFrozen(base, FIRST_TRAINED_BLOCK, head)


def save_model(model: PartlyFrozen, state, breeds, history, best_epoch, complete: bool):
    STAGE6_FILE.parent.mkdir(exist_ok=True)
    torch.save({"first_trained_block": model.first_trained_block, **state, "breeds": [b.name for b in breeds],
                "history": history, "best_epoch": best_epoch, "complete": complete,
                "info": {"base": "EfficientNet-B2 (ImageNet)", "trained_on": "Stanford Dogs train, 10,800 photos",
                         "epochs": EPOCHS, "lr": LR}}, STAGE6_FILE)


def cached_predictions(name, model, dataset, device):
    path = DATA / f"stage6_preds_{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["probs"], z["labels"]
    print(f"  predicting {name} ({len(dataset)} photos)...", flush=True)
    dataset.transform = EFFNET_B2.weights.transforms()
    probs, labels = predict_photos(model, dataset, device)
    np.savez(path, probs=probs, labels=labels)
    return probs, labels


def plot_training(history, best_epoch, baseline_val, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        s5.style(ax)
    ep = [h["epoch"] for h in history]
    ax1.plot(ep, [100 * h["val_accuracy"] for h in history], color=BLUE, linewidth=2, marker="o", markersize=5)
    ax1.axhline(100 * baseline_val, color=INK_2, linestyle=":", linewidth=1.5)
    ax1.text(ep[-1], 100 * baseline_val - 0.25, "epoch 0 = ImageNet's head (stage 4)", color=INK_2,
             fontsize=9, ha="right", va="top")
    ax1.set_xlabel("epoch (one pass over the 10,800 training photos)", color=INK_2)
    ax1.set_ylabel("validation accuracy (%)", color=INK_2)
    ax1.set_title("Validation accuracy per epoch", loc="left", color=INK)

    ax2.plot(ep[1:], [h["train_loss"] for h in history[1:]], color=BLUE, linewidth=2, marker="o", markersize=5,
             label="training photos (augmented, dropout on)")
    ax2.plot(ep, [h["val_loss"] for h in history], color=ORANGE, linewidth=2, marker="o", markersize=5,
             label="validation photos")
    ax2.axvline(best_epoch, color=INK, linestyle="--", linewidth=1)
    ax2.text(best_epoch + 0.08, 0.92, f"kept: epoch {best_epoch}", transform=ax2.get_xaxis_transform(),
             color=INK, fontsize=9)
    ax2.set_xlabel("epoch", color=INK_2)
    ax2.set_ylabel("loss (lower is better)", color=INK_2)
    ax2.set_title("Loss per epoch", loc="left", color=INK)
    ax2.legend(frameon=False, loc="lower left")
    for ax in (ax1, ax2):
        ax.set_xticks(ep)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def pair_analysis(models: dict, labels, names, pairs) -> list[dict]:
    """For each look-alike pair: per-breed accuracy and how often A is called B (and B called A), per model."""
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for a, b in pairs:
        ia, ib = idx[a], idx[b]
        row = {"a": a, "b": b}
        for key, probs in models.items():
            cm = confusion_matrix(probs, labels, len(names))
            na, nb = cm[ia].sum(), cm[ib].sum()
            row[key] = {"a_right": round(cm[ia, ia] / na, 4), "b_right": round(cm[ib, ib] / nb, 4),
                        "a_as_b": int(cm[ia, ib]), "b_as_a": int(cm[ib, ia]),
                        "pair_right": round((cm[ia, ia] + cm[ib, ib]) / (na + nb), 4)}
        rows.append(row)
    return rows


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    breeds = stanford_breeds(EFFNET_B2.categories)
    names = [b.name for b in breeds]

    print("Step 1: the same split as stage 5")
    train_all = stanford_dogs("train", breeds)
    labels = np.array([y for _, y in train_all.items])
    tr, va = stratified_split(labels, val_share=0.1, seed=SEED)
    train_ds = PhotoDataset([train_all.items[i] for i in tr], train_transform(EFFNET_B2.weights))
    val_ds = PhotoDataset([train_all.items[i] for i in va], EFFNET_B2.weights.transforms())
    print(f"  {len(train_ds)} training photos (augmented), {len(val_ds)} validation photos")

    print("Step 2: build the model")
    model = build_model(breeds)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  trainable: {n_train / 1e6:.1f}M weights (blocks {FIRST_TRAINED_BLOCK}-8 + head), "
          f"frozen: {n_frozen / 1e6:.1f}M")

    print(f"Step 3: train ({EPOCHS} epochs, learning rate {LR}, device {device})")
    saved = torch.load(STAGE6_FILE, weights_only=True) if STAGE6_FILE.exists() else None
    if saved and saved.get("complete") and "--retrain" not in sys.argv:
        print("  using the saved model (run with --retrain to train again)")
        history, best_epoch, train_minutes = saved["history"], saved["best_epoch"], saved["info"].get("minutes")
        model.load_trained_state(saved)
    else:
        for p in DATA.glob("stage6_preds_*.npz"):
            p.unlink()       # a new model makes old predictions invalid
        start = time.perf_counter()
        best_state, history, best_epoch = train_partly_frozen(
            model, train_ds, val_ds, device, epochs=EPOCHS, lr=LR, seed=SEED,
            on_epoch=lambda e, state, h, b: save_model(model, state, breeds, h, b, complete=False))
        train_minutes = round((time.perf_counter() - start) / 60, 1)
        model.load_trained_state(best_state)
        save_model(model, best_state, breeds, history, best_epoch, complete=True)
        saved = torch.load(STAGE6_FILE, weights_only=True)
        saved["info"]["minutes"] = train_minutes
        torch.save(saved, STAGE6_FILE)
    print(f"  kept epoch {best_epoch}: val accuracy {history[best_epoch]['val_accuracy']:.1%} "
          f"(epoch 0, ImageNet's head: {history[0]['val_accuracy']:.1%})")

    print("Step 4: test once, next to stages 4 and 5")
    sets = s5.photo_sets(breeds)
    P6 = {name: cached_predictions(name, model, sets[name][0], device)
          for name in ["stanford_test", "stage3_choose", "stage3_test"]}
    F = {name: s5.cached_features(name, sets[name][0], device) for name in ["stanford_test", "stage3_choose", "stage3_test"]}
    head5, _, _ = load_head(HEAD_FILE)
    test_y = F["stanford_test"][1]
    ox_y = F["stage3_test"][1]
    ox = ox_y >= 0
    s21 = np.isin(test_y, np.unique(ox_y[ox]))
    probs = {
        "imagenet_head": {n: s5.imagenet_breed_probs_from_features(F[n][0], breeds) for n in F},
        "stage5_head": {n: predict_probs(head5, F[n][0]) for n in F},
        "stage6": {n: P6[n][0] for n in P6},
    }
    assert (P6["stanford_test"][1] == test_y).all() and (P6["stage3_test"][1] == ox_y).all()
    test = {}
    for key, p in probs.items():
        st, o = p["stanford_test"], p["stage3_test"]
        test[key] = {"stanford_test": s5.report_set(st, test_y, breeds),
                     "stanford_test_21_breeds": {"photos": int(s21.sum()), "top1": topk_accuracy(st[s21], test_y[s21], 1),
                                                 "top3": topk_accuracy(st[s21], test_y[s21], 3)},
                     "oxford_test": s5.report_set(o, ox_y, breeds, ox)}
        r = test[key]
        print(f"  {key:13s} Stanford top-1 {r['stanford_test']['top1']:.1%}, Oxford top-1 {r['oxford_test']['top1']:.1%}, "
              f"ECE {r['stanford_test']['ece']:.3f} | {r['oxford_test']['ece']:.3f}")
    base_ox, p6_ox = probs["imagenet_head"]["stage3_test"], probs["stage6"]["stage3_test"]
    fixed = np.flatnonzero(ox & (base_ox.argmax(1) != ox_y) & (p6_ox.argmax(1) == ox_y))
    broken = np.flatnonzero(ox & (base_ox.argmax(1) == ox_y) & (p6_ox.argmax(1) != ox_y))
    print(f"  Oxford test: stage 6 fixes {len(fixed)} photos and breaks {len(broken)}")

    print("Step 5: error analysis of the look-alike pairs")
    pairs = [("Eskimo dog", "Siberian husky"), ("collie", "Border collie"),
             ("American Staffordshire terrier", "Staffordshire bullterrier"), ("English foxhound", "Walker hound"),
             ("miniature poodle", "toy poodle"), ("Siberian husky", "malamute")]
    pair_rows = pair_analysis({k: p["stanford_test"] for k, p in probs.items()}, test_y, names, pairs)
    for r in pair_rows:
        print(f"  {r['a']} / {r['b']}: both right " +
              ", ".join(f"{k} {r[k]['pair_right']:.0%}" for k in probs))

    print("Step 6: stage 3 thresholds and the ship rule")
    rows = {}
    for split, set_name in [("trainval", "stage3_choose"), ("test", "stage3_test")]:
        items = s3.list_photos(split)
        shares = s5.dog_shares(F[set_name][0])     # "is it a dog?" stays with ImageNet's head
        p = probs["stage6"][set_name]
        top3 = np.argsort(-p, axis=1)[:, :3]
        rows[split] = [{**it, "dog_share": float(s), "confidence": float(pp[t[0]]), "top3": [names[i] for i in t]}
                       for it, s, pp, t in zip(items, shares, p, top3)]
    dog_t, breed_t = s3.choose_thresholds(rows["trainval"])
    th = s3.report(rows["test"], dog_t, breed_t)
    s3.plot(rows["test"], dog_t, breed_t, f"{STAGE6_NAME}, test photos", ROOT / "docs" / "stage6_thresholds.png")
    green_base = json.loads((ROOT / "config" / "stage3_thresholds.json").read_text())["models"][EFFNET_B2.name]["test"]
    oxford_gain = test["stage6"]["oxford_test"]["top1"] - test["imagenet_head"]["oxford_test"]["top1"]
    ship = bool(oxford_gain >= MIN_OXFORD_GAIN and th["dogs_named_green"] >= green_base["dogs_named_green"])
    print(f"  dog threshold {dog_t}, breed threshold {breed_t}: 🟢 {th['dogs_named_green']:.1%} of dogs at "
          f"{th['green_accuracy']:.1%} (ImageNet's head: {green_base['dogs_named_green']:.1%})")
    print(f"  Oxford gain {100 * oxford_gain:+.1f} points -> ship: {ship}")
    (ROOT / "config" / "stage6_thresholds.json").write_text(json.dumps({
        "how": {"chosen_on": "Oxford-IIIT Pets trainval + pizza_steak_sushi train (same rules as stage 3)",
                "tested_on": "Oxford-IIIT Pets test + pizza_steak_sushi test",
                "keep_dogs": s3.KEEP_DOGS, "target_accuracy": s3.TARGET_ACCURACY},
        "models": {STAGE6_NAME: {"dog_threshold": dog_t, "breed_threshold": breed_t,
                                 "plot": "docs/stage6_thresholds.png", "test": th}}}, indent=2) + "\n")

    print("Step 7: save")
    paths = [str(p.relative_to(ROOT)) for p, _ in sets["stage3_test"][0].items]
    rng = np.random.default_rng(SEED)
    examples = ([{"path": paths[i], "note": "fixed"} for i in rng.choice(fixed, min(4, len(fixed)), replace=False)]
                + [{"path": paths[i], "note": "broken"} for i in rng.choice(broken, min(2, len(broken)), replace=False)])
    plot_training(history, best_epoch, history[0]["val_accuracy"], ROOT / "docs" / "stage6_training.png")
    s5.plot_breed_changes(test["imagenet_head"]["stanford_test"]["per_breed"], test["stage6"]["stanford_test"]["per_breed"],
                          ROOT / "docs" / "stage6_breed_changes.png",
                          title="Biggest changes per breed: stage 4 → stage 6")
    out = {"model": STAGE6_NAME,
           "setup": {"first_trained_block": FIRST_TRAINED_BLOCK, "trainable_weights": n_train, "frozen_weights": n_frozen,
                     "epochs": EPOCHS, "lr": LR, "train_minutes": train_minutes},
           "history": history, "best_epoch": best_epoch, "test": test, "pairs": pair_rows,
           "oxford_fixed": len(fixed), "oxford_broken": len(broken), "examples": examples,
           "thresholds": {"dog": dog_t, "breed": breed_t, "test": th},
           "ship_rule": {"min_oxford_gain": MIN_OXFORD_GAIN, "oxford_gain": oxford_gain,
                         "green_needed": green_base["dogs_named_green"], "green": th["dogs_named_green"], "ship": ship}}
    (ROOT / "reports" / "stage6_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Saved models/stage6_b2.pth, reports/stage6_report.json, config/stage6_thresholds.json, docs/stage6_*.png")


if __name__ == "__main__":
    main()
