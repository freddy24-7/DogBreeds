"""Stage 2: real predictions from a pre-trained ImageNet model. See README_stage2.md."""
import gradio as gr
from PIL import Image

from dogbreeds.predictors import EFFNET_B0, EFFNET_B2

ID = "stage2"
TITLE = "Stage 2: Pre-trained ImageNet model"

MODELS = {p.name: p for p in [EFFNET_B2, EFFNET_B0]}


def classify(image: Image.Image | None, model_name: str):
    """Returns: top breeds, a one-line summary, and the top ImageNet classes overall."""
    if image is None:
        return {}, "", {}
    analysis = MODELS[model_name].analyse(image, top_k=5)
    summary = (
        f"**{analysis.dog_share:.0%}** of the model's probability went to dog breeds "
        f"(the other {1 - analysis.dog_share:.0%} went to the 882 other ImageNet classes). "
        f"Prediction took {analysis.seconds * 1000:.0f} ms."
    )
    return dict(analysis.breeds), summary, dict(analysis.imagenet_top)


def build() -> None:
    gr.Markdown(
        "**What's new:** a real model. EfficientNet was trained by others on ImageNet "
        "(1.3 million photos, 1000 classes, 118 of them dog breeds). We use it exactly as it is, "
        "with **no training** by us, and keep only the dog breeds. "
        "Try a dog photo, then a cat or a person, and look at the dog share."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo of a dog", sources=["upload", "webcam", "clipboard"])
            model = gr.Dropdown(choices=list(MODELS), value=EFFNET_B2.name, label="Model")
        with gr.Column():
            breeds = gr.Label(num_top_classes=3, label="Top 3 breeds")
            summary = gr.Markdown()
            with gr.Accordion("Under the hood: top 5 out of all 1000 ImageNet classes", open=False):
                imagenet_top = gr.Label(num_top_classes=5, show_label=False)

    outputs = [breeds, summary, imagenet_top]
    image.change(classify, inputs=[image, model], outputs=outputs, api_name="stage2_classify")
    model.change(classify, inputs=[image, model], outputs=outputs, api_name=False)
