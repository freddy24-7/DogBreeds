"""Stage 7: where did the model look? Heatmaps (CAM / Grad-CAM). See README_stage7.md."""
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from dogbreeds.decision import decide, load_thresholds
from dogbreeds.heatmap import class_maps, feature_maps, fix_corner, model_view, overlay, to_heat
from dogbreeds.predictors import EFFNET_B2

ID = "stage7"
TITLE = "Stage 7: Where did it look?"

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "stage7_report.json"
FOR, VERSUS = "Evidence for this answer", "Evidence for the 1st answer rather than the 2nd"


def analyse(image: Image.Image | None):
    """The slow part, done once per photo: the model's answer and its 1,408 feature maps."""
    if image is None:
        return None, gr.update(choices=[], value=None), ""
    a = EFFNET_B2.analyse(image, top_k=3)
    d = decide(a, load_thresholds(EFFNET_B2.name))
    x, crop = model_view(image, EFFNET_B2.weights)
    A = feature_maps(EFFNET_B2.model, x)
    answers = [a.imagenet_top[0][0]] if d.status == "no_dog" else [b for b, _ in d.breeds]
    state = {"A": A, "crop": crop, "answers": answers}
    return state, gr.update(choices=answers, value=answers[0]), d.message


def render(state, answer: str | None, mode: str, raw: bool):
    """The fast part: one heatmap from the stored feature maps (no need to run the model again)."""
    if state is None or answer is None:
        return None, None, ""
    A = state["A"] if raw else fix_corner(state["A"])
    cats = EFFNET_B2.categories
    if mode == VERSUS and len(state["answers"]) >= 2:
        first, second = state["answers"][:2]
        maps = class_maps(EFFNET_B2.model, A, [cats.index(first), cats.index(second)])[0]
        cam, what = maps[0] - maps[1], f"where the photo looks more like a **{first}** than a **{second}**"
    else:
        cam, what = class_maps(EFFNET_B2.model, A, [cats.index(answer)])[0, 0], f"where it saw **{answer}**"
    heat = to_heat(cam, state["crop"].size[0])
    note = (f"Orange = {what}. The map is a 9 × 9 grid (one cell is 32 × 32 pixels), smoothed to the photo's size. "
            + ("**Raw map:** note the top-left corner, a model artifact that lights up on any photo."
               if raw else "The top-left cell is repaired (a model artifact, see below)."))
    if heat.max() == 0:
        note = "No evidence anywhere in the photo for this answer: the whole map is negative."
    return state["crop"], overlay(state["crop"], heat), note


def results_table(r: dict) -> str:
    lines = ["| Oxford test photos | Photos | Pet's share of the picture | Heat on the pet | "
             "Hottest spot on the pet | Same, raw map (with the artifact) |", "|---|---|---|---|---|---|"]
    for name, g in r["groups"].items():
        lines.append(f"| {name} | {g['photos']} | {g['pet_area']:.0%} | **{g['heat_on_pet']:.0%}** | "
                     f"**{g['pointing_game']:.0%}** | {g['raw_heat_on_pet']:.0%} · {g['raw_pointing_game']:.0%} |")
    return "\n".join(lines)


def build() -> None:
    gr.Markdown(
        "**What's new:** a heatmap of **where the model looked** to give its answer. It explains the model the "
        "app ships (EfficientNet-B2 with ImageNet's head, stages 2–4). Below the demo: the heatmaps checked "
        "against the true outlines of 3,600 pets, and a flaw in the model they revealed."
    )
    report = json.loads(REPORT.read_text()) if REPORT.exists() else None
    examples = [str(ROOT / e["photo"]) for e in report["examples"]] if report else []
    examples = [p for p in examples if Path(p).exists()]
    state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"])
            if examples:
                gr.Examples(examples, inputs=image, label="Oxford test photos: 4 typical, 3 small dogs (right), "
                                                          "3 small dogs (wrong)")
        with gr.Column(scale=2):
            message = gr.Markdown()
            with gr.Row():
                answer = gr.Radio([], label="Explain which answer?")
                mode = gr.Radio([FOR, VERSUS], value=FOR, label="Map")
            raw = gr.Checkbox(False, label="Show the raw map (with the top-left corner artifact)")
            with gr.Row():
                seen = gr.Image(label="What the model sees (the centre square)", interactive=False, height=300)
                heat = gr.Image(label="Where it looked", interactive=False, height=300)
            note = gr.Markdown()

    image.change(analyse, image, [state, answer, message], api_name="stage7_analyse").then(
        render, [state, answer, mode, raw], [seen, heat, note], api_name=False)
    for control in (answer, mode, raw):
        control.change(render, [state, answer, mode, raw], [seen, heat, note], api_name=False)

    if report is None:
        gr.Markdown("*No measurement yet: run `uv run python scripts/stage7_heatmaps.py`.*")
        return
    gr.Markdown("### Does it look at the pet? Measured on Oxford's outlines")
    gr.Markdown(
        "Oxford Pets gives the outline of every pet. For each test photo we check how much of the heatmap for "
        "the model's answer falls on the pet, and whether the hottest spot is on it (the *pointing game*). A map "
        "that ignored the pet and spread evenly would score the pet's share of the picture, about 50%."
    )
    gr.Markdown(results_table(report))
    acc = report["breed_accuracy"]
    gr.Markdown(
        "**Yes, it looks at the pet**, for right and wrong answers alike. Wrong answers aren't caused by looking "
        "at the background: they're the look-alikes from stage 4. When the heat does spill off the pet, it's "
        "usually a small dog far away, wider than one cell of the coarse 9 × 9 grid."
    )
    gr.Image(str(ROOT / "docs" / "stage7_heat_on_pet.png"), show_label=False, interactive=False,
             buttons=["fullscreen"])
    gr.Markdown("### A flaw the heatmaps revealed: the top-left corner")
    gr.Markdown(
        "In the raw maps, the **top-left cell** is often the hottest, on any photo. It stays top-left when the "
        "photo is mirrored, and it even appears on a plain grey image, so it's a property of the model, not of "
        "the photo. These weights were trained in TensorFlow and ported to PyTorch; the two pad image edges "
        "differently, which is the most likely cause. It even nudges answers: with the cell repaired, breed "
        f"accuracy on these 21 breeds goes from {acc['as_is']:.1%} to {acc['corner_repaired']:.1%}. Below: "
        "the photo, the raw map, and the repaired map."
    )
    gr.Image(str(ROOT / report["corner_example"]["image"]), show_label=False, interactive=False,
             buttons=["fullscreen"])
    gr.Gallery([(str(ROOT / e["image"]), f"{e['kind']}: {e['truth']} → said {e['said']} "
                                         f"({e['heat_on_pet']:.0%} of heat on the dog)")
                for e in report["examples"]], label="Examples: the photo the model sees, and its heatmap",
               columns=2, height="auto")
