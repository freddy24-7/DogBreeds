"""
The DogBreeds web app.

Run from the project folder with:
    uv run python app.py
then open http://localhost:7870 in your browser.

Each stage of the project has its own tab (see dogbreeds/stages/), so you can
compare how the app behaved at every stage. The newest stage opens first.
The tabs only talk to models through the Predictor interface in
dogbreeds/predictors.py.
"""
import gradio as gr

from dogbreeds.stages import STAGES

with gr.Blocks(title="DogBreeds") as demo:
    gr.Markdown(
        "# 🐕 DogBreeds\n"
        "Upload a photo of a dog and see which breed the model thinks it is. "
        "Each tab is one stage of the project; the newest is open."
    )
    with gr.Tabs(selected=STAGES[-1].ID):
        for stage in STAGES:
            with gr.Tab(stage.TITLE, id=stage.ID):
                stage.build()


if __name__ == "__main__":
    # Port 7870, not Gradio's default 7860, which another project (a Docker container) already uses.
    demo.launch(server_port=7870)
