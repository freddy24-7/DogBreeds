"""Stage 1: the app with a fake model. See README.md and the Predictor interface."""
import gradio as gr
from PIL import Image

from dogbreeds.predictors import DUMMY

ID = "stage1"
TITLE = "Stage 1: Dummy model"


def classify(image: Image.Image | None) -> dict[str, float]:
    if image is None:
        return {}
    # gr.Label wants {label: probability}; it draws the bars itself.
    return dict(DUMMY.predict(image, top_k=3))


def build() -> None:
    gr.Markdown(
        "**What's new:** the app itself: upload a photo, get a top-3 list. "
        "The model is a dummy that guesses at random and doesn't look at the photo. "
        "Upload the same photo twice and you get different breeds."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo of a dog", sources=["upload", "webcam", "clipboard"])
        with gr.Column():
            result = gr.Label(num_top_classes=3, label="Top 3 breeds")
    image.change(classify, inputs=image, outputs=result, api_name="stage1_classify")
