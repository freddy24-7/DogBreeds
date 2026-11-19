"""The text of the app's About and Code tabs."""

REPO = "https://github.com/freddy24-7/DogBreeds"

ABOUT = """
## What this is

**DogBreeds** recognises a dog's breed from a photo. It's also a learning project: it was built in **nine
stages**, and each stage adds one idea from machine learning engineering. Every stage is still in the app, so
you can pick any stage in the **Demo** tab and see how the app behaved at that point.

## How it was built

| Stage | What it adds | What we found |
|---|---|---|
| 1. Dummy model | The app and a `Predictor` interface, with random guesses | The app can be built and tested before there is any model |
| 2. Pre-trained model | A real ImageNet model (EfficientNet), no training by us | ImageNet already knows 118 dog breeds |
| 3. Knowing when it doesn't know | "I don't see a dog" and "not sure which breed" | Measured thresholds: 97.5% of dogs accepted, 95.7% of cats turned away |
| 4. Model report | Accuracy on 8,580 Stanford Dogs test photos | 95.1% on Stanford Dogs but 90.3% on Oxford's photos: the test set overlaps the model's training data |
| 5. Our own head | A new final layer, trained on Stanford Dogs | Better calibrated, but not more accurate: not shipped |
| 6. Fine-tuning | The last backbone blocks retrained, with data augmentation | Slightly worse (89.6% vs 90.3%): not shipped, by a rule set before training |
| 7. Where did it look? | Heatmaps of where the model found its evidence | 84% of the heat falls on the dog; the heatmaps also exposed a flaw in the model's top-left corner |
| 8. Dogs like yours | The most similar of 12,000 photos, with no breed head at all | 87% of the closest photos share the breed; the search found copied photos filed under two breeds |
| 9. Ready to ship | A self-contained 120 MB package of the app, measured as a server would run it | It needs 566 MB of memory, more than free hosting plans give, so it isn't online; the README shows how to host it |

The model the app uses for breeds is **EfficientNet-B2 with ImageNet's own head** (stages 2–4). Stages 5 and 6
trained alternatives, but neither beat it on the measures we set in advance, so they weren't shipped.

## How to read the answers

- 🟢 **A breed:** the model is confident, and at that confidence it was right about 95% of the time.
- 🟡 **Not sure:** it shows its top 3; the right breed was among them about 94% of the time.
- 🔴 **No dog:** most of the model's probability is on things that aren't dogs.
- Realistic accuracy on everyday photos is **about 90%**. Look-alike breeds (Norfolk and Norwich terrier,
  husky and Eskimo dog) cause most of the mistakes.

## Data and models

- **Stanford Dogs** (Khosla et al., 2011): 20,580 photos of 120 breeds, taken from ImageNet. Used for
  training (stages 5–6), testing (stage 4) and as the photo gallery (stage 8).
- **Oxford-IIIT Pet** (Parkhi et al., 2012): photos of 25 dog and 12 cat breeds, with pet outlines. Used for
  the thresholds (stage 3), the fairer accuracy check (stage 4) and checking the heatmaps (stage 7).
- **EfficientNet-B0 and B2** (Tan & Le, 2019), with ImageNet weights from torchvision.
- A small food photo set (pizza, steak, sushi; a subset of Food-101) for the "no dog" check.

The datasets are for non-commercial research and education.
"""

CODE = f"""
## The code

All the code is on GitHub: **[{REPO.removeprefix("https://")}]({REPO})**

It's written to be read. Each stage has a `README_stageN.md` that explains the steps and concepts, with the
measurements behind every decision.

| Where | What's in it |
|---|---|
| `app.py` | This app: the header, the stage picker, and the About and Code tabs |
| `dogbreeds/predictors.py` | The `Predictor` interface and every model behind it |
| `dogbreeds/stages/` | One module per stage; each builds its own page |
| `dogbreeds/decision.py` | The 🟢 / 🟡 / 🔴 rules (stage 3) |
| `dogbreeds/data.py`, `evaluate.py` | Datasets and metrics (stage 4) |
| `dogbreeds/finetune.py` | Training our own head and fine-tuning (stages 5–6) |
| `dogbreeds/heatmap.py` | Heatmaps (stage 7) |
| `dogbreeds/similar.py` | Similarity search (stage 8) |
| `scripts/` | The measurements that produce `config/`, `reports/` and `docs/`; `stage9_deploy.py` packages the app for a server |
| `tests/` | The tests, including a contract every predictor must pass |

**Run it yourself:**

```bash
git clone {REPO}.git
cd DogBreeds
uv run python app.py      # then open http://localhost:7870
uv run pytest             # run the tests
```

Stages 1–3 work straight away. The later stages need the datasets and the files their scripts make
(trained weights, the search index); each stage's README says how to make them.

Built with PyTorch, torchvision and Gradio.
"""
