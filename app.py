"""
The DogBreeds web app.

Run from the project folder with:
    uv run python app.py
then open http://localhost:7870 in your browser.

Layout:
    header      the project name and tagline on a colour gradient
    Demo tab    a "Choose stage" dropdown; each stage of the project has its own page
                (see dogbreeds/stages/), so you can compare how the app behaved at every
                stage. The newest stage opens first.
    About tab   what the project is, the results so far, and the data and models used
    Code tab    the GitHub repository and where to find each stage's code

The stage pages only talk to models through the Predictor interface in
dogbreeds/predictors.py.
"""
import gradio as gr

from dogbreeds import about
from dogbreeds.stages import STAGES

# (label shown in the dropdown, stage id), newest first
STAGE_CHOICES = [(stage.TITLE, stage.ID) for stage in reversed(STAGES)]

HEADER = """
<div class="db-header">
  <h1>🐕 DogBreeds</h1>
  <p>Built with love for dogs and machine learning</p>
</div>
"""

# The gradient runs from burnt orange through raspberry and purple to indigo. Every colour in
# it has a contrast of at least 5:1 with white text, so the header stays readable everywhere.
CSS = """
.db-header {
  background: linear-gradient(120deg, #c2410c 0%, #be185d 38%, #9333ea 72%, #4f46e5 100%);
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 4px;
  box-shadow: 0 6px 24px rgba(79, 70, 229, 0.18);
}
.db-header h1 { color: #ffffff; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; }
.db-header p { color: #ffffff; margin: 6px 0 0; font-size: 1.05rem; font-weight: 500; opacity: 0.95; }
@media (max-width: 640px) {
  .db-header { padding: 20px 18px; }
  .db-header h1 { font-size: 1.7rem; }
}
.db-picker-hint { align-self: center; color: var(--body-text-color-subdued); }
"""


def show_stage(stage_id: str):
    """Shows the chosen stage's page and hides the others."""
    return [gr.update(visible=stage.ID == stage_id) for stage in STAGES]


with gr.Blocks(title="DogBreeds") as demo:
    gr.HTML(HEADER)
    with gr.Tabs():
        with gr.Tab("Demo", id="demo"):
            with gr.Row(equal_height=True):
                picker = gr.Dropdown(STAGE_CHOICES, value=STAGES[-1].ID, label="Choose stage", scale=1,
                                     min_width=280)
                with gr.Column(scale=2, min_width=240):          # on a phone it wraps below the picker
                    gr.Markdown("Each stage is one step of the project, and its page shows the app as it was "
                                "then. The newest stage is selected. The **About** tab tells the whole story.",
                                elem_classes="db-picker-hint")
            pages = []
            for stage in STAGES:
                with gr.Column(visible=stage is STAGES[-1]) as page:
                    stage.build()
                pages.append(page)
            picker.change(show_stage, picker, pages, api_name=False)
        with gr.Tab("About", id="about"):
            gr.Markdown(about.ABOUT)
        with gr.Tab("Code", id="code"):
            gr.Markdown(about.CODE)


if __name__ == "__main__":
    # Port 7870, not Gradio's default 7860, which another project (a Docker container) already uses.
    demo.launch(server_port=7870, css=CSS)
