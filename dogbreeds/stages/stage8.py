"""Stage 8: dogs that look like yours. Similarity search with embeddings. See README_stage8.md."""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image

from dogbreeds.data import stanford_breeds
from dogbreeds.decision import decide, load_thresholds
from dogbreeds.predictors import EFFNET_B2
from dogbreeds.similar import Index, embed

ID = "stage8"
TITLE = "Stage 8: Dogs like yours"

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "stage8_report.json"
INDEX_FILE = ROOT / "models" / "stage8_index.npz"


@lru_cache(maxsize=1)
def index_and_names() -> tuple[Index, list[str]]:
    """Loaded on first use: the saved gallery (12,000 photos) and the 120 breed names."""
    return Index.load(INDEX_FILE), [b.name for b in stanford_breeds(EFFNET_B2.categories)]


def search(image: Image.Image | None, k: int):
    if image is None:
        return [], "", ""
    index, names = index_and_names()
    report = json.loads(REPORT.read_text())
    threshold = report["unusual"]["threshold"]

    d = decide(EFFNET_B2.analyse(image, top_k=3), load_thresholds(EFFNET_B2.name))
    sims, pos = index.search(embed(EFFNET_B2, [image]), k=int(k))
    sims, pos = sims[0], pos[0]
    neighbours = [(str(ROOT / index.paths[p]), f"{names[index.labels[p]]} · {s:.2f}") for s, p in zip(sims, pos)]

    breed, n = Counter(names[index.labels[p]] for p in pos).most_common(1)[0]
    lines = [f"**The model (stage 3):** {d.message}"]
    if sims[0] >= threshold:
        lines.append(f"**The neighbours:** {n} of the {len(pos)} most similar gallery photos are **{breed}**.")
    else:                                     # a vote among far-away photos would mean nothing
        lines.append(f"⚠️ **Nothing in the gallery looks much like this photo.** The closest is {sims[0]:.2f}; "
                     f"99% of dog photos have one above {threshold:.2f}. These are the closest *dogs*, "
                     "not necessarily similar ones.")
    note = (f"Similarity: 1 = the same direction in embedding space; the gallery is the 12,000 Stanford Dogs "
            f"training photos. Search time: about {report['index']['ms_per_search']:.1f} ms.")
    return neighbours, "\n\n".join(lines), note


def pct(x: float) -> str:
    return f"{x:.1%}"


def build() -> None:
    gr.Markdown(
        "**What's new:** no breed head at all. The backbone turns every photo into 1,408 numbers (an "
        "**embedding**); photos that look alike get similar numbers. Upload a dog and see the 12,000 Stanford "
        "training photos that look most like it. The same search also found copied photos in the dataset."
    )
    if not (REPORT.exists() and INDEX_FILE.exists()):
        gr.Markdown("*No index yet: run `uv run python scripts/stage8_similar.py`.*")
        return
    report = json.loads(REPORT.read_text())
    examples = [str(ROOT / p) for p in report["examples"] if (ROOT / p).exists()]

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Photo", sources=["upload", "webcam", "clipboard"])
            k = gr.Slider(4, 16, value=8, step=4, label="How many similar photos")
            if examples:
                gr.Examples(examples, inputs=image, label="Oxford test photos (not in the gallery); the last is a cat")
        with gr.Column(scale=2):
            message = gr.Markdown()
            gallery = gr.Gallery(label="Most similar gallery photos (breed · similarity)", columns=4, height="auto")
            note = gr.Markdown()
    for trigger in (image.change, k.release):
        trigger(search, [image, k], [gallery, message, note], api_name=False)

    gr.Markdown("### How good is the search?")
    rows = ["| How embeddings are compared | Numbers per photo | Same breed among the 5 closest | "
            "k-NN breed accuracy |", "|---|---|---|---|"]
    for v in report["variants"]:
        mark = " ✅ chosen" if v["key"] == report["chosen"] else ""
        rows.append(f"| {v['label']}{mark} | {v['numbers_per_photo']} | {pct(v['precision_at_5'])} | "
                    f"{pct(v['knn_accuracy'])} |")
    gr.Markdown(
        "Measured on validation photos (10% of the training set, searched against the other 90%). The rule, "
        f"set beforehand: the **smallest** index within 0.5 points of the best. PCA keeps 64 of 1,408 numbers "
        f"and the index shrinks to {report['index']['megabytes']:.0f} MB.\n\n" + "\n".join(rows))
    t = report["test"]
    gr.Markdown(
        "**On the test photos** (used once):\n\n"
        "| Test photos | Same breed: closest photo | Among the 5 closest | k-NN accuracy (10 neighbours vote) | "
        "ImageNet's head (stage 4) |\n|---|---|---|---|---|\n"
        f"| Stanford Dogs test (8,580) | {pct(t['stanford']['precision_at_1'])} | {pct(t['stanford']['precision_at_5'])} | "
        f"{pct(t['stanford']['knn_accuracy'])} | {pct(t['stanford']['imagenet_head_accuracy'])} |\n"
        f"| … without the {report['copies']['same_shoot']} that have a same-shoot photo in the gallery | "
        f"{pct(t['stanford_no_same_shoot']['precision_at_1'])} | {pct(t['stanford_no_same_shoot']['precision_at_5'])} | "
        f"{pct(t['stanford_no_same_shoot']['knn_accuracy'])} | – |\n"
        f"| Oxford dogs (another collection) | {pct(t['oxford_dogs']['precision_at_1'])} | "
        f"{pct(t['oxford_dogs']['precision_at_5'])} | {pct(t['oxford_dogs']['knn_accuracy'])} | "
        f"{pct(t['oxford_dogs']['imagenet_head_accuracy'])} |\n\n"
        "A search with **no training at all** gets within 3–4 points of a trained head. The head is still "
        "better at naming the breed; the search is better at *showing* why."
    )
    gr.Image(str(ROOT / report["images"]["neighbours"]), show_label=False, interactive=False,
             buttons=["fullscreen"])

    c = report["copies"]
    gr.Markdown("### What the search found in the dataset: copies")
    gr.Markdown(
        f"For every test photo we looked up its closest **training** photo. **{c['copies']}** test photos are "
        f"also in the training set (similarity ≥ 0.99, the same picture), and **{c['copies_with_a_different_breed']}** "
        f"of them have a **different breed** in each set. Within the training set, {c['train_photos_with_a_copy_in_train']} "
        f"photos have a copy, {c['of_which_different_breed']} under another breed. The conflicting labels are the "
        "look-alike pairs from stage 4 (Norfolk / Norwich terrier, Blenheim / Japanese spaniel, wire-haired fox / "
        f"Lakeland terrier): even the dataset's makers confused them. Another {c['same_shoot'] - c['copies']} test "
        "photos come from the same photo shoot as a training photo (similarity ≥ 0.88). Leaving those out barely "
        "changes the scores above, so they don't inflate the results much."
    )
    gr.Image(str(ROOT / "docs" / "stage8_copies.png"), show_label=False, interactive=False, buttons=["fullscreen"])
    gr.Image(str(ROOT / report["images"]["copies_with_two_breeds"]), show_label=False, interactive=False,
             buttons=["fullscreen"])

    u = report["unusual"]
    gr.Markdown("### Can similarity tell a dog from a cat?")
    gr.Markdown(
        f"Partly. Stage 3's photos: if the closest gallery photo is below **{u['threshold']:.2f}** (a threshold "
        f"that keeps 99% of dogs, set on stage 3's 'choose' photos), we warn \"nothing like it\". On the test "
        f"photos that flags {pct(u['groups']['dog']['flagged'])} of dogs, {pct(u['groups']['cat']['flagged'])} "
        f"of cats and {pct(u['groups']['food']['flagged'])} of food. A search **always** returns something, so a "
        "check like this is needed. But stage 3's \"is it a dog?\" rule does better (it keeps 97.5% of dogs and "
        "turns away 95.7% of cats and all the food), so the app keeps that as the main check."
    )
    gr.Image(str(ROOT / "docs" / "stage8_similarity_by_group.png"), show_label=False, interactive=False,
             buttons=["fullscreen"])
