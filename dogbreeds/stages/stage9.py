"""Stage 9: ready to ship, and what it would take to host. See README_stage9.md."""
import json
from pathlib import Path

import gradio as gr

ID = "stage9"
TITLE = "Stage 9: Ready to ship"

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "stage9_report.json"
GITHUB = "https://github.com/freddy24-7/DogBreeds"


def build() -> None:
    if not REPORT.exists():
        gr.Markdown("*Not measured yet: run `uv run python scripts/stage9_deploy.py`.*")
        return
    r = json.loads(REPORT.read_text())
    url = r.get("space_url")
    online = f" It's online at **[{url.removeprefix('https://')}]({url})**." if url else ""
    gr.Markdown(
        "**What's new:** no new model. This stage packs the app so it runs **on its own, on any server**, and "
        "measures what a server would need. It isn't hosted online: every host that can run it asks for a "
        f"subscription or a credit card, so this stage stops at \"ready to ship\".{online} "
        f"[README_stage9.md]({GITHUB}/blob/main/README_stage9.md) explains how to put it online."
    )

    gr.Markdown("### What's in the package")
    rows = ["| Part | Size |", "|---|---|"]
    rows += [f"| {name} | {mb:.1f} MB |" for name, mb in r["parts_megabytes"].items()]
    rows.append(f"| **Total** | **{r['bundle_megabytes']:.0f} MB** in {r['files']:,} files |")
    g = r["gallery"]
    gr.Markdown(
        "\n".join(rows) + "\n\n"
        f"- **The gallery** of stage 8 is the biggest part. The original {g['photos']:,} photos are "
        f"{g['original_megabytes']:.0f} MB; shrunk to {g['thumbnail_pixels']} pixels (they're shown at about "
        "150), they're a fifth of that. The search index doesn't change: it never used the photos, only "
        "their embeddings.\n"
        "- **Not packed:** " + "; ".join(r["not_bundled"]) + ".\n"
        "- **The same versions everywhere:** `requirements.txt` has the exact versions from `uv.lock` (with "
        f"PyTorch's smaller CPU-only build), and Gradio {r['gradio']}."
    )

    m = r.get("measured")
    if m:
        gr.Markdown("### What a server needs")
        rows = ["| Model | Time per photo |", "|---|---|"]
        rows += [f"| {name} | {s:.2f} s |" for name, s in m["seconds_per_photo"].items()]
        gr.Markdown(
            "Measured by running the package on its own, in a fresh Python, on this laptop's CPU (no GPU):\n\n"
            f"- **Start-up:** {m['start_seconds']:.0f} seconds to load the libraries and build every page.\n"
            f"- **Memory:** **{m['peak_memory_megabytes']} MB** at its peak, with every stage's model loaded. "
            "This is the number that rules out most free hosting plans: they stop at 512 MB, so the app needs "
            "a server with **1 GB**.\n\n" + "\n".join(rows)
        )

    gr.Markdown(
        "### How to put it online\n"
        "One command builds the package (`deploy/space/`), measures it, and, with `--upload`, sends it to a "
        "Hugging Face Space:\n\n"
        "```bash\nuv run python scripts/stage9_deploy.py --upload USER/DogBreeds\n```\n\n"
        "Hugging Face only hosts Gradio apps for **PRO** accounts (a paid subscription). The package also runs "
        "on any server or container with Python 3.12 and 1 GB of memory: install `requirements.txt` and run "
        f"`python app.py`. [README_stage9.md]({GITHUB}/blob/main/README_stage9.md) compares the options."
    )
