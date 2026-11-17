"""
Stage 4: the model report. The fixed benchmark every later model must beat.

Run from the project folder (about 25 minutes the first time, seconds afterwards):
    uv run python scripts/stage4_report.py

For both ImageNet models it measures, on photos the app has never been tuned on:
  - Stanford Dogs test split: 8,580 photos, 120 breeds (the main benchmark)
  - Oxford-IIIT Pets test dogs: the 21 breeds Stanford also has (the leakage check)
  - data/my_photos/: your own photos, if you added any (the most honest test)

Writes reports/stage4_report.json (read by the app) and plots in docs/.
The model's raw predictions are saved in data/ so reruns are quick; delete
data/stage4_preds_*.npz to really rerun the models.
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
from dogbreeds.data import DATA, OXFORD_TO_IMAGENET, my_photos, oxford_dogs, stanford_breeds, stanford_dogs  # noqa: E402
from dogbreeds.evaluate import imagenet_breed_probs, model_report, topk_accuracy  # noqa: E402
from dogbreeds.predictors import EFFNET_B0, EFFNET_B2  # noqa: E402

COLORS = {"b2": "#2a78d6", "b0": "#eb6834"}  # validated default palette, slots 1 and 2
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
MODELS = {"b2": EFFNET_B2, "b0": EFFNET_B0}


def predictions(key, predictor, set_name, dataset, breeds, device):
    cache = DATA / f"stage4_preds_{key}_{set_name}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["probs"], z["labels"], float(z["seconds"])
    print(f"  running {predictor.name} on {set_name} ({len(dataset)} photos)...", flush=True)
    probs, labels, seconds = imagenet_breed_probs(predictor, dataset, breeds, device)
    np.savez_compressed(cache, probs=probs, labels=labels, seconds=seconds)
    return probs, labels, seconds


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot_calibration(reports, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.6), facecolor=SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax.text(0.62, 0.55, "perfectly calibrated", color=INK_2, rotation=37, fontsize=9)
    for key, rep in reports.items():
        bins = [b for b in rep["calibration"] if b["count"] >= 20]  # skip tiny groups: too noisy
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=COLORS[key], linewidth=2,
                marker="o", markersize=5, label=f"{MODELS[key].name.split(' (')[0]} (ECE {rep['ece']:.3f})")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("how sure the model is (top probability)", color=INK_2)
    ax.set_ylabel("how often it's right", color=INK_2)
    ax.set_title("Calibration on Stanford Dogs test", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_hardest(rep, title, path, n=15):
    items = sorted(rep["per_breed"].items(), key=lambda kv: kv[1])[:n][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=SURFACE)
    style(ax)
    ax.barh([k for k, _ in items], [100 * v for _, v in items], color=COLORS["b2"], height=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(100 * v - 1.5, i, f"{100 * v:.0f}%", va="center", ha="right", color="white", fontsize=9)
    ax.axvline(100 * rep["top1"], color=INK, linestyle="--", linewidth=1)
    ax.text(100 * rep["top1"], n - 0.3, f"average over all 120 breeds: {100 * rep['top1']:.0f}%",
            color=INK, fontsize=9, ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n + 0.4)
    ax.set_xlabel("top-1 accuracy on that breed's test photos (%)", color=INK_2)
    ax.set_title(title, loc="left", color=INK)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    breeds = stanford_breeds(EFFNET_B0.categories)
    sets = {"stanford_test": stanford_dogs("test", breeds), "oxford_test": oxford_dogs("test", breeds)}
    mine = my_photos(breeds)
    if mine:
        sets["my_photos"] = mine
    print(f"{len(breeds)} breeds; " + ", ".join(f"{k}: {len(v)} photos" for k, v in sets.items()))

    oxford_breeds = {b.index for b in breeds if b.name in set(OXFORD_TO_IMAGENET.values())}
    out = {"breeds": [b.name for b in breeds], "models": {}}
    stanford_reports = {}
    for key, predictor in MODELS.items():
        print(f"\n{predictor.name}")
        res = {}
        for set_name, ds in sets.items():
            probs, labels, seconds = predictions(key, predictor, set_name, ds, breeds, device)
            res[set_name] = model_report(probs, labels, breeds)
            res[set_name]["seconds_per_photo_batched"] = seconds
            if set_name == "stanford_test":
                # Examples for the most-confused pairs: one wrong photo each (from the Stanford test set)
                pred = probs.argmax(axis=1)
                name_to_i = {b.name: b.index for b in breeds}
                for pair in res[set_name]["confused_pairs"]:
                    t, p = name_to_i[pair["true"]], name_to_i[pair["predicted"]]
                    i = int(np.flatnonzero((labels == t) & (pred == p))[0])
                    pair["example"] = str(ds.items[i][0].relative_to(ROOT))
                # The same 21 breeds as Oxford, for a like-for-like leakage comparison
                mask = np.isin(labels, list(oxford_breeds))
                res["stanford_test_21_breeds"] = {"photos": int(mask.sum()),
                                                  "top1": topk_accuracy(probs[mask], labels[mask], 1),
                                                  "top3": topk_accuracy(probs[mask], labels[mask], 3)}
                stanford_reports[key] = res[set_name]
            r = res[set_name]
            print(f"  {set_name:14s} top-1 {r['top1']:.1%}  top-3 {r['top3']:.1%}  ECE {r['ece']:.3f}  ({r['photos']} photos)")
        s21, ox = res["stanford_test_21_breeds"], res["oxford_test"]
        print(f"  leakage check (21 breeds): Stanford top-1 {s21['top1']:.1%} vs Oxford {ox['top1']:.1%}")
        plot_hardest(res["stanford_test"], f"Hardest 15 breeds, {predictor.name.split(' (')[0]}",
                     ROOT / "docs" / f"stage4_hardest_{key}.png")
        out["models"][predictor.name] = res

    plot_calibration(stanford_reports, ROOT / "docs" / "stage4_calibration.png")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "stage4_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nSaved reports/stage4_report.json and plots in docs/")


if __name__ == "__main__":
    main()
