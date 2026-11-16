"""
The DogBreeds web app.

Run from the project folder with:
    uv run python app.py
then open http://localhost:7870 in your browser.

The app only talks to predictors through `predictor.predict(image)`
(see dogbreeds/predictors.py), so this file barely changes when we add models.
"""
import gradio as gr
from PIL import Image

from dogbreeds.predictors import PREDICTORS


def classify(image: Image.Image | None, model_name: str) -> dict[str, float]:
    """Called by Gradio when a photo is uploaded or the model changes."""
    if image is None:
        return {}
    predictions = PREDICTORS[model_name].predict(image, top_k=3)
    # gr.Label wants {label: probability}; it draws the bars itself.
    return dict(predictions)


with gr.Blocks(title="DogBreeds") as demo:
    gr.Markdown(
        "# 🐕 DogBreeds\n"
        "Upload a photo of a dog and see which breed the model thinks it is."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Photo of a dog", sources=["upload", "webcam", "clipboard"])
            model = gr.Dropdown(choices=list(PREDICTORS), value=next(iter(PREDICTORS)), label="Model")
        with gr.Column():
            result = gr.Label(num_top_classes=3, label="Top 3 breeds")
            gr.Markdown(
                "*Stage 1: the only model is a dummy that guesses at random. "
                "It doesn't look at the photo at all. Real models arrive in stage 2.*"
            )

    # Predict again whenever the photo or the chosen model changes.
    image.change(classify, inputs=[image, model], outputs=result)
    model.change(classify, inputs=[image, model], outputs=result)


if __name__ == "__main__":
    # Port 7870, not Gradio's default 7860, which another project (a Docker container) already uses.
    demo.launch(server_port=7870)
