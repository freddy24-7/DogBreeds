"""Stage 4: the model report, the benchmark every later model must beat. See README_stage4.md."""
import json
from pathlib import Path

import gradio as gr

from dogbreeds.predictors import EFFNET_B0, EFFNET_B2

ID = "stage4"
TITLE = "Stage 4: Model report"

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "stage4_report.json"
KEYS = {EFFNET_B2.name: "b2", EFFNET_B0.name: "b0"}


def short(name: str) -> str:
    return name.split(" (")[0]


def summary_table(models: dict) -> str:
    names = list(models)
    rows = [
        ("**Stanford Dogs test** (8,580 photos, 120 breeds)", None, None),
        ("Top-1 accuracy", "stanford_test", "top1"),
        ("Top-3 accuracy", "stanford_test", "top3"),
        ("Top-5 accuracy", "stanford_test", "top5"),
        ("Average confidence", "stanford_test", "mean_confidence"),
        ("Calibration error (ECE, lower is better)", "stanford_test", "ece"),
        ("**Leakage check: the same 21 breeds**", None, None),
        ("Stanford Dogs, top-1 (photos ImageNet may have seen)", "stanford_test_21_breeds", "top1"),
        ("Oxford Pets, top-1 (photos it likely hasn't seen)", "oxford_test", "top1"),
    ]
    if all("my_photos" in m for m in models.values()):
        rows += [("**Your own photos**", None, None), ("Top-1 accuracy", "my_photos", "top1"),
                 ("Top-3 accuracy", "my_photos", "top3")]
    lines = ["| | " + " | ".join(short(n) for n in names) + " |", "|---|" + "---|" * len(names)]
    for label, part, metric in rows:
        if part is None:
            lines.append(f"| {label} |" + " |" * len(names))
            continue
        vals = [models[n][part][metric] for n in names]
        fmt = (lambda v: f"{v:.3f}") if metric == "ece" else (lambda v: f"{v:.1%}")
        lines.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
    return "\n".join(lines)


def leakage_text(models: dict) -> str:
    m = models[EFFNET_B2.name]
    s, o = m["stanford_test_21_breeds"]["top1"], m["oxford_test"]["top1"]
    return (f"On the same 21 breeds, B2 scores **{s:.1%}** on Stanford Dogs but **{o:.1%}** on Oxford Pets: "
            f"a gap of **{100 * (s - o):.1f} points**. Stanford Dogs was built from ImageNet photos, so the model "
            "has probably seen many of these 'test' photos during its training. Part of the gap may also come "
            "from the photos themselves (Oxford's are more everyday snapshots). Either way: "
            "**the Stanford number flatters the model**; Oxford's is closer to what users will see.")


def details(model_name: str, models: dict):
    rep = models[model_name]["stanford_test"]
    key = KEYS[model_name]
    pairs = rep["confused_pairs"]
    table = ["| True breed | Mistaken for | Count | Share of its photos |", "|---|---|---|---|"]
    table += [f"| {p['true']} | {p['predicted']} | {p['count']} | {p['share']:.0%} |" for p in pairs]
    gallery = [(str(ROOT / p["example"]), f"{p['true']} → taken for {p['predicted']}")
               for p in pairs if (ROOT / p["example"]).exists()]
    return str(ROOT / "docs" / f"stage4_hardest_{key}.png"), "\n".join(table), gallery


def build() -> None:
    gr.Markdown(
        "**What's new:** a proper measurement instead of impressions. Both models were tested on "
        "**8,580 Stanford Dogs photos of 120 breeds** they were never tuned on, plus a leakage check. "
        "This report is the **baseline**: from stage 5 on, every new model must beat it on the same photos."
    )
    if not REPORT.exists():
        gr.Markdown("*No report yet: run `uv run python scripts/stage4_report.py`.*")
        return
    models = json.loads(REPORT.read_text())["models"]

    gr.Markdown("### Summary")
    gr.Markdown(summary_table(models))
    gr.Markdown("### Can we trust the test? Leakage\n" + leakage_text(models))

    gr.Markdown("### Is \"80% sure\" right 80% of the time? Calibration")
    with gr.Row():
        gr.Image(str(ROOT / "docs" / "stage4_calibration.png"), show_label=False, interactive=False,
                 buttons=["fullscreen"])
        gr.Markdown(
            "Each dot groups photos by how sure the model was. On the dashed line, confidence matches reality. "
            "**Below the line** = overconfident (says 90%, right less often). **Above** = underconfident. "
            "ECE is the average distance from the line, weighted by how many photos are in each group."
        )

    gr.Markdown("### Where it goes wrong")
    model = gr.Dropdown(choices=list(KEYS), value=EFFNET_B2.name, label="Model")
    plot0, table0, gallery0 = details(EFFNET_B2.name, models)   # what's shown before you pick a model
    with gr.Row():
        hardest = gr.Image(plot0, show_label=False, interactive=False, buttons=["fullscreen"])
        pairs = gr.Markdown(table0)
    gallery = gr.Gallery(gallery0, label="One example of each confusion (click to enlarge)", columns=5,
                         height="auto")

    model.change(lambda name: details(name, models), model, [hardest, pairs, gallery], api_name=False)
