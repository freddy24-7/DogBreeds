"""Stage 5: our own head on a frozen backbone (transfer learning). See README_stage5.md."""
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from dogbreeds.decision import decide, load_thresholds
from dogbreeds.predictors import EFFNET_B2, FINETUNED_B2, HEAD_FILE

ID = "stage5"
TITLE = "Stage 5: Our own head"

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "stage5_report.json"
STAGE3_CONFIG = ROOT / "config" / "stage3_thresholds.json"
STAGE5_CONFIG = ROOT / "config" / "stage5_thresholds.json"


def compare(image: Image.Image | None):
    """One backbone pass, both heads, each with its own measured stage 3 thresholds."""
    if image is None:
        return "", {}, "", {}, ""
    result = FINETUNED_B2.compare(image, top_k=5)
    d_theirs = decide(result["imagenet"], load_thresholds(EFFNET_B2.name, STAGE3_CONFIG))
    d_ours = decide(result["ours"], load_thresholds(FINETUNED_B2.name, STAGE5_CONFIG))
    a = result["ours"]
    details = (f"Dog share **{a.dog_share:.0%}** (from ImageNet's head, the same for both) · "
               f"one pass through the backbone, both heads: {a.seconds * 1000:.0f} ms")
    return (d_theirs.message, {} if d_theirs.status == "no_dog" else dict(d_theirs.breeds[:3]),
            d_ours.message, {} if d_ours.status == "no_dog" else dict(d_ours.breeds[:3]), details)


def pct(v: float) -> str:
    return f"{v:.1%}"


def results_table(report: dict) -> str:
    t = report["test"]
    th3 = json.loads(STAGE3_CONFIG.read_text())["models"][EFFNET_B2.name]["test"]
    th5 = report["thresholds"]["test"]
    rows = [
        ("**Stanford Dogs test** (8,580 photos, 120 breeds)", None, None),
        ("Top-1 accuracy", lambda m: pct(m["stanford_test"]["top1"]), None),
        ("Top-3 accuracy", lambda m: pct(m["stanford_test"]["top3"]), None),
        ("Calibration error (ECE, lower is better)", lambda m: f"{m['stanford_test']['ece']:.3f}", None),
        ("**Oxford check** (2,086 unseen photos, 21 breeds)", None, None),
        ("Top-1 accuracy", lambda m: pct(m["oxford_test"]["top1"]), None),
        ("Top-3 accuracy", lambda m: pct(m["oxford_test"]["top3"]), None),
        ("Calibration error (ECE)", lambda m: f"{m['oxford_test']['ece']:.3f}", None),
        ("**In the app** (stage 3 rules, Oxford test photos)", None, None),
        ("Dogs that get a 🟢 breed name", None, (pct(th3["dogs_named_green"]), pct(th5["dogs_named_green"]))),
        ("Accuracy of the 🟢 names", None, (pct(th3["green_accuracy"]), pct(th5["green_accuracy"]))),
    ]
    lines = ["| | ImageNet's head (stage 4 baseline) | Our head |", "|---|---|---|"]
    for label, fn, fixed in rows:
        if fn is None and fixed is None:
            lines.append(f"| {label} | | |")
        elif fixed:
            lines.append(f"| {label} | {fixed[0]} | {fixed[1]} |")
        else:
            lines.append(f"| {label} | {fn(t['imagenet_head'])} | {fn(t['our_head'])} |")
    return "\n".join(lines)


def verdict(report: dict) -> str:
    t = report["test"]
    theirs, ours = t["imagenet_head"], t["our_head"]
    st = 100 * (ours["stanford_test"]["top1"] - theirs["stanford_test"]["top1"])
    ox = 100 * (ours["oxford_test"]["top1"] - theirs["oxford_test"]["top1"])
    green3 = json.loads(STAGE3_CONFIG.read_text())["models"][EFFNET_B2.name]["test"]["dogs_named_green"]
    green5 = report["thresholds"]["test"]["dogs_named_green"]
    better = ox >= 1.0 and green5 >= green3
    return "\n".join([
        f"- **Accuracy:** {st:+.1f} points on Stanford Dogs, {ox:+.1f} on the Oxford check. On the unseen Oxford "
        f"photos our head fixes {report['oxford_fixed']} answers and breaks {report['oxford_broken']}: "
        + ("a real gain." if ox >= 1.0 else "**no real gain**, the differences are within the noise."),
        f"- **Calibration:** much better, ECE {theirs['oxford_test']['ece']:.3f} → {ours['oxford_test']['ece']:.3f} "
        "on Oxford: when our head says \"90% sure\", it is right about 90% of the time. Most likely, our plain training "
        "dropped the caution built into ImageNet's own training (label smoothing).",
        f"- **In the app:** to keep 🟢 answers 95% right, our head needs a breed threshold of "
        f"{report['thresholds']['breed']:.2f}, and then only **{green5:.0%}** of dogs get a 🟢 name, against "
        f"**{green3:.0%}** with ImageNet's head. Its confidence is more honest, but it separates right from "
        "wrong answers *less* well.",
        "",
        ("**Verdict: our head is better. It becomes the default.**" if better else
         "**Verdict: don't ship it.** The app keeps ImageNet's head. A model that is no more accurate and "
         "answers 🟢 less often isn't an improvement, however nice its calibration. That's a normal, useful "
         "outcome: the baseline from stage 4 did its job."),
    ])


def experiments_table(report: dict) -> str:
    lines = ["| Setting | Best epoch | Validation accuracy | |", "|---|---|---|---|"]
    for e in report["experiments"]:
        chosen = "✅ chosen" if e["key"] == report["chosen"] else ""
        lines.append(f"| {e['label']} | {e['best_epoch']} | {e['val_accuracy']:.1%} | {chosen} |")
    return "\n".join(lines)


def build() -> None:
    gr.Markdown(
        "**What's new:** for the first time we **train** something. The ImageNet model's backbone (its "
        "\"eyes\") stays frozen; we replaced its 1,000-class head with **our own head for the 120 breeds**, "
        "trained on 10,800 Stanford Dogs photos. Same photos, same tests as stage 4: did it beat the baseline?"
    )
    if not (REPORT.exists() and HEAD_FILE.exists()):
        gr.Markdown("*Not trained yet: run `uv run python scripts/stage5_train_head.py`.*")
        return
    report = json.loads(REPORT.read_text())
    examples = [str(ROOT / e["path"]) for e in report["examples"] if (ROOT / e["path"]).exists()]

    gr.Markdown("### Try it: both heads on the same photo")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"])
            if examples:
                gr.Examples(examples, inputs=image,
                            label="Unseen Oxford photos where the heads disagree: 4 fixed by our head, then 2 it broke")
        with gr.Column(scale=1):
            gr.Markdown("**ImageNet's head** (stages 2–4)")
            msg_theirs = gr.Markdown()
            label_theirs = gr.Label(num_top_classes=3, show_label=False)
        with gr.Column(scale=1):
            gr.Markdown("**Our own head** (stage 5)")
            msg_ours = gr.Markdown()
            label_ours = gr.Label(num_top_classes=3, show_label=False)
    details = gr.Markdown()
    image.change(compare, image, [msg_theirs, label_theirs, msg_ours, label_ours, details], api_name="stage5_compare")

    gr.Markdown("### Did it beat the baseline?")
    gr.Markdown(results_table(report))
    gr.Markdown(verdict(report))
    gr.Image(str(ROOT / "docs" / "stage5_breed_changes.png"), show_label=False, interactive=False,
             buttons=["fullscreen"])

    gr.Markdown("### How it was trained")
    gr.Markdown(
        "Every setting was trained for 30 epochs on the **training** photos and compared on the **validation** "
        "photos (10 per breed, set aside). The test photos above played no part in choosing. From each run "
        "we keep the epoch with the lowest validation loss (*early stopping*)."
    )
    gr.Markdown(experiments_table(report))
    gr.Image(str(ROOT / "docs" / "stage5_training.png"), show_label=False, interactive=False, buttons=["fullscreen"])
    with gr.Accordion("Our head's own stage 3 thresholds", open=False):
        th = report["thresholds"]
        gr.Markdown(f"Measured with stage 3's rules on Oxford photos: dog threshold **{th['dog']:.2f}** "
                    f"(unchanged, it comes from ImageNet's head) and breed threshold **{th['breed']:.2f}**.")
        gr.Image(str(ROOT / "docs" / "stage5_thresholds.png"), show_label=False, interactive=False,
                 buttons=["fullscreen"])
