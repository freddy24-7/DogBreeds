"""Stage 3: saying "I don't see a dog" and "I'm not sure". See README_stage3.md."""
from pathlib import Path

import gradio as gr
from PIL import Image

from dogbreeds.decision import Thresholds, decide, load_thresholds
from dogbreeds.predictors import EFFNET_B0, EFFNET_B2, ImageNetAnalysis

ID = "stage3"
TITLE = "Stage 3: Knowing when it doesn't know"

ROOT = Path(__file__).resolve().parents[2]
MODELS = {p.name: p for p in [EFFNET_B2, EFFNET_B0]}
PLOTS = {EFFNET_B2.name: ROOT / "docs" / "stage3_thresholds_b2.png",
         EFFNET_B0.name: ROOT / "docs" / "stage3_thresholds_b0.png"}

# A few photos from the test split to click on, if the datasets are downloaded (they're not in git).
EXAMPLES = [ROOT / "data" / p for p in [
    "oxford-iiit-pet/images/pug_194.jpg",               # 🟢 clear breed
    "oxford-iiit-pet/images/beagle_195.jpg",            # 🟡 beagle or a similar hound
    "oxford-iiit-pet/images/Bengal_192.jpg",            # 🔴 a cat, correctly rejected
    "pizza_steak_sushi/test/pizza/1152100.jpg",         # 🔴 food, correctly rejected
    "oxford-iiit-pet/images/shiba_inu_1.jpg",           # mistake: a real dog, but "no dog"
    "oxford-iiit-pet/images/Sphynx_211.jpg",            # mistake: a hairless cat taken for a dog
]]
EXAMPLES = [str(p) for p in EXAMPLES if p.exists()]


def analyse(image: Image.Image | None, model_name: str) -> ImageNetAnalysis | None:
    """The slow part (running the model). Its result is kept, so moving a slider doesn't rerun it."""
    return None if image is None else MODELS[model_name].analyse(image, top_k=5)


def render(analysis: ImageNetAnalysis | None, dog_t: float, breed_t: float):
    """The fast part: apply the thresholds to a stored analysis."""
    if analysis is None:
        return "", {}, "", {}
    d = decide(analysis, Thresholds(dog=dog_t, breed=breed_t))
    breed_conf = d.breeds[0][1]
    details = (
        f"Dog share **{analysis.dog_share:.0%}** (threshold {dog_t:.0%}) · "
        f"breed confidence **{breed_conf:.0%}** (threshold {breed_t:.0%}) · "
        f"{analysis.seconds * 1000:.0f} ms"
    )
    breeds = {} if d.status == "no_dog" else dict(d.breeds)
    return d.message, breeds, details, dict(analysis.imagenet_top)


def thresholds_for(model_name: str):
    t = load_thresholds(model_name)
    return t.dog, t.breed, str(PLOTS[model_name])


def build() -> None:
    gr.Markdown(
        "**What's new:** the app can now say *\"I don't see a dog\"* (🔴) or "
        "*\"a dog, but I'm not sure which breed\"* (🟡) instead of always naming 3 breeds. "
        "The two thresholds behind this were **measured** on about 7,700 photos of dogs, cats and food. "
        "Try a cat or your lunch, and play with the thresholds under *Experiment*."
    )
    default = EFFNET_B2.name
    dog_t0, breed_t0, plot0 = thresholds_for(default)
    analysis = gr.State(None)  # the last analysis, kept between events

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"])
            model = gr.Dropdown(choices=list(MODELS), value=default, label="Model")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=image, label="Test photos: green, yellow, cat, food, then two mistakes")
        with gr.Column():
            message = gr.Markdown()
            breeds = gr.Label(num_top_classes=3, label="Breed, if it's a dog")
            details = gr.Markdown()
            with gr.Accordion("Experiment: move the thresholds", open=False):
                gr.Markdown("Defaults are the measured values for this model. "
                            "Higher = stricter: fewer mistakes, but more real dogs rejected or left unnamed.")
                dog_t = gr.Slider(0, 1, value=dog_t0, step=0.01, label="Dog threshold (minimum dog share)")
                breed_t = gr.Slider(0, 1, value=breed_t0, step=0.01, label="Breed threshold (minimum breed confidence)")
                plot = gr.Image(value=plot0, label="How the thresholds were chosen (test photos)",
                                interactive=False, buttons=["fullscreen"])
            with gr.Accordion("Under the hood: top 5 out of all 1000 ImageNet classes", open=False):
                imagenet_top = gr.Label(num_top_classes=5, show_label=False)

    outputs = [message, breeds, details, imagenet_top]
    # New photo: run the model, then apply the thresholds.
    image.change(analyse, [image, model], analysis, api_name="stage3_analyse").then(
        render, [analysis, dog_t, breed_t], outputs, api_name=False)
    # New model: load its thresholds and plot, rerun the model, apply.
    model.change(thresholds_for, model, [dog_t, breed_t, plot], api_name=False).then(
        analyse, [image, model], analysis, api_name=False).then(
        render, [analysis, dog_t, breed_t], outputs, api_name=False)
    # Slider moved: only reapply the thresholds (instant, the model doesn't run again).
    for slider in (dog_t, breed_t):
        slider.change(render, [analysis, dog_t, breed_t], outputs, api_name=False)
