"""Stage 6: fine-tuning the last backbone blocks, with augmentation and error analysis. See README_stage6.md."""
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from dogbreeds.decision import decide, load_thresholds
from dogbreeds.predictors import EFFNET_B2, FINETUNED6_B2, STAGE6_FILE

ID = "stage6"
TITLE = "Stage 6: Fine-tuning"

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "stage6_report.json"
STAGE3_CONFIG = ROOT / "config" / "stage3_thresholds.json"
STAGE5_REPORT = ROOT / "reports" / "stage5_report.json"
STAGE6_CONFIG = ROOT / "config" / "stage6_thresholds.json"
MODELS = {"imagenet_head": "ImageNet's head (stage 4)", "stage5_head": "Our head (stage 5)",
          "stage6": "Fine-tuned (stage 6)"}


def compare(image: Image.Image | None):
    if image is None:
        return "", {}, "", {}, ""
    result = FINETUNED6_B2.compare(image, top_k=5)
    d_theirs = decide(result["imagenet"], load_thresholds(EFFNET_B2.name, STAGE3_CONFIG))
    d_ours = decide(result["ours"], load_thresholds(FINETUNED6_B2.name, STAGE6_CONFIG))
    a = result["ours"]
    details = (f"Dog share **{a.dog_share:.0%}** (from ImageNet's blocks and head, the same for both) · "
               f"blocks 0–6 run once for both: {a.seconds * 1000:.0f} ms")
    return (d_theirs.message, {} if d_theirs.status == "no_dog" else dict(d_theirs.breeds[:3]),
            d_ours.message, {} if d_ours.status == "no_dog" else dict(d_ours.breeds[:3]), details)


def pct(v: float) -> str:
    return f"{v:.1%}"


def results_table(report: dict) -> str:
    t = report["test"]
    green = {"imagenet_head": json.loads(STAGE3_CONFIG.read_text())["models"][EFFNET_B2.name]["test"],
             "stage6": report["thresholds"]["test"]}
    if STAGE5_REPORT.exists():
        green["stage5_head"] = json.loads(STAGE5_REPORT.read_text())["thresholds"]["test"]
    rows = [
        ("**Stanford Dogs test** (8,580 photos)", None),
        ("Top-1 accuracy", lambda k: pct(t[k]["stanford_test"]["top1"])),
        ("Top-3 accuracy", lambda k: pct(t[k]["stanford_test"]["top3"])),
        ("Calibration error (ECE)", lambda k: f"{t[k]['stanford_test']['ece']:.3f}"),
        ("**Oxford check** (2,086 unseen photos, 21 breeds)", None),
        ("Top-1 accuracy", lambda k: pct(t[k]["oxford_test"]["top1"])),
        ("Top-3 accuracy", lambda k: pct(t[k]["oxford_test"]["top3"])),
        ("Calibration error (ECE)", lambda k: f"{t[k]['oxford_test']['ece']:.3f}"),
        ("**In the app** (stage 3 rules, Oxford test photos)", None),
        ("Dogs that get a 🟢 breed name", lambda k: pct(green[k]["dogs_named_green"]) if k in green else "–"),
        ("Accuracy of the 🟢 names", lambda k: pct(green[k]["green_accuracy"]) if k in green else "–"),
    ]
    lines = ["| | " + " | ".join(MODELS.values()) + " |", "|---|" + "---|" * len(MODELS)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(k) if fn else "" for k in MODELS) + " |")
    return "\n".join(lines)


def verdict(report: dict) -> str:
    r = report["ship_rule"]
    t = report["test"]
    st = 100 * (t["stage6"]["stanford_test"]["top1"] - t["imagenet_head"]["stanford_test"]["top1"])
    lines = [
        f"- **Accuracy:** {st:+.1f} points on Stanford Dogs and **{100 * r['oxford_gain']:+.1f} points on the "
        f"Oxford check** (the rule asks for at least +{100 * r['min_oxford_gain']:.0f}). On the unseen Oxford photos "
        f"it fixes {report['oxford_fixed']} answers and breaks {report['oxford_broken']}.",
        f"- **In the app:** {r['green']:.0%} of dogs get a 🟢 name (the rule asks for at least {r['green_needed']:.0%}, "
        "what ImageNet's head achieves).",
        "",
        "**Verdict: ship it.** Both parts of the rule are met, so the fine-tuned model is better where it counts."
        if r["ship"] else
        "**Verdict: don't ship it.** The rule, written down before training, isn't met. The app keeps "
        "ImageNet's head.",
    ]
    return "\n".join(lines)


def pairs_table(report: dict) -> str:
    lines = ["| Look-alike pair (A / B) | Both breeds right (stage 4 → 5 → 6) | A called B · B called A (stage 4 → 6) |",
             "|---|---|---|"]
    for p in report["pairs"]:
        right = " → ".join(f"{p[k]['pair_right']:.0%}" for k in MODELS)
        swaps = (f"{p['imagenet_head']['a_as_b']} · {p['imagenet_head']['b_as_a']} → "
                 f"{p['stage6']['a_as_b']} · {p['stage6']['b_as_a']}")
        lines.append(f"| {p['a']} / {p['b']} | {right} | {swaps} |")
    return "\n".join(lines)


def build() -> None:
    gr.Markdown(
        "**What's new:** stage 5 showed that a new head on *frozen* features can't beat ImageNet's. Now the "
        "**last two blocks of the backbone** are trained too, on real photos with **data augmentation**, so the "
        "features themselves can adapt to dog breeds. The rule for shipping it was written down before training."
    )
    if not (REPORT.exists() and STAGE6_FILE.exists()):
        gr.Markdown("*Not trained yet: run `uv run python scripts/stage6_finetune.py` (about an hour).*")
        return
    report = json.loads(REPORT.read_text())
    examples = [str(ROOT / e["path"]) for e in report["examples"] if (ROOT / e["path"]).exists()]

    gr.Markdown("### Try it: ImageNet's head vs the fine-tuned model")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"])
            if examples:
                gr.Examples(examples, inputs=image,
                            label="Unseen Oxford photos where they disagree: first fixed by fine-tuning, then broken")
        with gr.Column(scale=1):
            gr.Markdown("**ImageNet's head** (stages 2–4)")
            msg_theirs = gr.Markdown()
            label_theirs = gr.Label(num_top_classes=3, show_label=False)
        with gr.Column(scale=1):
            gr.Markdown("**Fine-tuned** (stage 6)")
            msg_ours = gr.Markdown()
            label_ours = gr.Label(num_top_classes=3, show_label=False)
    details = gr.Markdown()
    image.change(compare, image, [msg_theirs, label_theirs, msg_ours, label_ours, details], api_name="stage6_compare")

    gr.Markdown("### Did it beat the baseline?")
    gr.Markdown(results_table(report))
    gr.Markdown(verdict(report))

    gr.Markdown("### Error analysis: the look-alike pairs")
    gr.Markdown(
        "Stage 4 found that most mistakes are between look-alike breeds, and stage 5's head only moved the "
        "border between them. Did fine-tuning learn to tell them apart? \"Both breeds right\" is the accuracy "
        "over all test photos of the two breeds together, so moving the border doesn't improve it, only "
        "telling them apart does."
    )
    gr.Markdown(pairs_table(report))
    gr.Image(str(ROOT / "docs" / "stage6_breed_changes.png"), show_label=False, interactive=False,
             buttons=["fullscreen"])

    gr.Markdown("### How it was trained")
    s = report["setup"]
    gr.Markdown(
        f"Blocks 0–{s['first_trained_block'] - 1} frozen ({s['frozen_weights'] / 1e6:.1f}M weights), blocks "
        f"{s['first_trained_block']}–8 and the head trained ({s['trainable_weights'] / 1e6:.1f}M weights), "
        f"{s['epochs']} epochs at learning rate {s['lr']:g}, on the same 10,800 training photos as stage 5, "
        f"randomly cropped, flipped and recoloured every epoch. Training took {s['train_minutes']} minutes "
        f"on the M1. Kept: **epoch {report['best_epoch']}** (lowest validation loss)."
    )
    gr.Image(str(ROOT / "docs" / "stage6_training.png"), show_label=False, interactive=False, buttons=["fullscreen"])
    with gr.Accordion("The fine-tuned model's own stage 3 thresholds", open=False):
        th = report["thresholds"]
        gr.Markdown(f"Dog threshold **{th['dog']:.2f}** (from ImageNet's branch, unchanged) and breed threshold "
                    f"**{th['breed']:.2f}**, measured with stage 3's rules on Oxford photos.")
        gr.Image(str(ROOT / "docs" / "stage6_thresholds.png"), show_label=False, interactive=False,
                 buttons=["fullscreen"])
