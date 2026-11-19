# Stage 9: ready to ship

**Goal:** get the app ready to run **on a server**, not just on this laptop, and find out what that server needs.

**Result:**
- `scripts/stage9_deploy.py` builds a self-contained package of the app (`deploy/space/`, 120 MB) and measures it.
- The "Stage 9" page shows the package's contents, its speed and memory, and how to put it online.
- This README explains how to deploy the package, step by step, on the main kinds of host.

**The outcome, up front:**
1. **The package runs on its own.** It has everything the nine stage pages need, and nothing else. It started in a fresh Python, outside the project, and every stage answered.
2. **It needs about 570 MB of memory**, so a server with **1 GB**. Most free hosting plans stop at 512 MB, which rules them out.
3. **It isn't online.** We tried Hugging Face Spaces first. Creating the Space failed with *"hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"*. The other hosts with enough memory also need a subscription or a credit card. So this stage stops at "ready to ship", with a guide for when you want to pay for hosting.

## Why "deploying" is more than uploading

On this laptop the app can reach everything: the full datasets (1.8 GB), the training scripts, the cached ImageNet weights, our virtual environment. A server has none of it. Deploying means answering three questions:

1. **What does the app actually need at runtime?** Ship exactly that.
2. **Will it behave the same?** Same library versions, same files at the same paths.
3. **What hardware does it need?** Memory, CPU, and how long it takes to start.

## Step 1: choosing what to ship

| Part | Size | Why |
|---|---|---|
| Code (`app.py`, `dogbreeds/`) | 0.1 MB | the app itself |
| Settings, reports and plots (`config/`, `reports/`, `docs/`) | 3.6 MB | the thresholds measured in stages 3 and 7, and every number and chart the pages show |
| Our model files (`models/`) | 20.5 MB | the stage 5 head, the stage 6 fine-tuned blocks, the stage 8 index |
| Example photos (50) | 4.6 MB | the photos the pages offer to click on, copied as they are |
| Gallery thumbnails (12,000) | 90.9 MB | the photos stage 8 shows as "dogs like yours" |
| **Total** | **120 MB** | |

**Left out on purpose:**
- **The ImageNet weights.** torchvision downloads them the first time a model is used, just as it did on this laptop. They'd add about 50 MB, and torchvision's copy is the official one.
- **The datasets.** The app never reads them. It only shows a few example photos and the stage 8 gallery.
- **`scripts/` and `tests/`.** They're for building the project, not running it.

### The gallery: shrinking 456 MB to 91 MB

Stage 8 shows photos from a gallery of 12,000. The originals are 456 MB, nearly four times everything else combined. But the page shows them as tiles about 150 pixels wide, so we shrink each one to at most **192 pixels** on its longest side. That's a fifth of the size and looks the same on screen.

The search itself doesn't change. The index stores each photo's **embedding** (64 numbers), not the photo, so smaller photos give exactly the same neighbours.

### Same paths, so the code doesn't change

Every file keeps its path from the project folder. For example, `data/stanford-dogs/Images/n02085620-Chihuahua/n02085620_10621.jpg` is at that same path inside the package. So the app's code runs unchanged, with no "if on a server" branches.

The script checks this before building: it collects every photo path the pages or reports refer to, and stops if any is missing (`referenced_photos()`). A test checks the same thing (`tests/test_deploy.py`).

## Step 2: the same versions everywhere

A model can give different answers with a different PyTorch, and a Gradio page can break with a different Gradio. So the package's `requirements.txt` pins the **exact versions in `uv.lock`**:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.15.0
torchvision==0.29.2
numpy==2.5.5
pillow==13.0.0
scipy==1.18.1
```

- **The CPU-only build of PyTorch** (the `--extra-index-url` line). Cheap servers have no GPU, and the default PyTorch build includes GPU support worth gigabytes. The CPU build is a fraction of that.
- **Gradio isn't in the list.** A Hugging Face Space installs it itself, at the version in the package's `README.md` (`sdk_version: 6.32.0`). On any other host, add `gradio==6.32.0`.
- **Python 3.12.** Hosts support it widely. This laptop runs 3.14, so this is the one difference we couldn't test locally.

The script reads the versions from the installed packages. So when we upgrade something, the next build picks it up, and the test `test_requirements_pin_the_versions_we_test_with` keeps them in step.

## Step 3: measuring what a server needs

The script runs the package the way a server would: a **fresh Python process, inside `deploy/space/`**, with none of the project folder around it. It starts the app, runs every stage's model on one photo, and measures:

| | Measured on this laptop's CPU |
|---|---|
| Start-up (libraries loaded, every page built) | 7 s |
| Peak memory, with every stage's model loaded | **566 MB** |
| Stage 3 (ImageNet EfficientNet-B2) | 0.28 s per photo |
| Stage 5 (our head) | 0.28 s per photo |
| Stage 6 (fine-tuned blocks) | 0.25 s per photo |
| Stage 7 (the dog/not-dog check, then the breed) | 0.55 s per photo |
| Stage 8 (similarity search) | 0.74 s per photo |

**How to read it:**
- **Memory decides the host.** About 400 MB of the 566 MB is the start-up alone (PyTorch, Gradio, the pages). Each model loads only when a page first uses it, and adds the rest. On a 512 MB server the app would start, then be killed when a visitor tried a second stage. With **1 GB** there's room to spare.
- **Speed is fine on a CPU.** Every answer takes under a second, so a GPU isn't needed. A shared server CPU is slower than this laptop's, so expect perhaps 2–3 times longer.
- **The first visit is slower.** On a host that sleeps when nobody uses it, the first visitor waits for the server to start, the libraries to load (7 s here) and the ImageNet weights to download.

## Step 4: putting it online

### Option A: a Hugging Face Space (needs PRO)

This is what the script was built for. A Space is a git repository that Hugging Face runs for you; the `README.md` at the top of the package holds its settings (the Gradio version, Python version, which file to run).

1. Subscribe to [Hugging Face PRO](https://huggingface.co/pro). Free accounts can only host static pages.
2. Create a **Write** access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then log in:
   ```bash
   uv run hf auth login
   ```
3. Build, measure and upload in one command:
   ```bash
   uv run python scripts/stage9_deploy.py --upload YOUR-USERNAME/DogBreeds
   ```
   This creates the Space if it doesn't exist and uploads the 12,000-file package. The Space then installs `requirements.txt` and starts `app.py`, which reads the port the Space gives it (`GRADIO_SERVER_PORT`, 7860).
4. Watch the build log on the Space's page. The first build takes a few minutes. When it says *Running*, try each stage.

### Option B: any server or container

The package is a plain Python app, so it runs anywhere with Python 3.12 and 1 GB of memory: a cloud VM, a container service (Google Cloud Run, Fly.io, Railway, Render's paid plans), or a serverless GPU/CPU platform like Modal. The steps are the same everywhere:

```bash
cd deploy/space
pip install -r requirements.txt gradio==6.32.0
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8080 python app.py
```

- `GRADIO_SERVER_NAME=0.0.0.0` makes Gradio accept visitors from outside the machine. By default it only listens to the machine itself.
- `GRADIO_SERVER_PORT` is whatever port the host expects. `app.py` uses 7870 if it isn't set.
- The server needs internet access once, to download the ImageNet weights.

For a container host, the same three lines go in a `Dockerfile`, starting from the `python:3.12-slim` image.

### What the hosts cost (checked November 2026)

| Host | Memory on the free plan | Fits the app? |
|---|---|---|
| Hugging Face Spaces | Gradio apps need PRO | ✅ with PRO |
| Render, Koyeb | 512 MB | ❌ too little (566 MB measured) |
| Google Cloud Run | pay per use, with a monthly free allowance | ✅ but needs a billing account with a card |
| Modal | monthly credits, then pay per use | ✅ but not free |

Prices and free plans change often, so check the host's own pricing page before choosing.

### Making it fit a smaller server

If a 512 MB plan were the only option, these would bring the memory down, roughly in order of effort:
1. **Drop stages from the online version.** Stages 2–3 use two ImageNet models; keeping only the newest pages loads fewer models.
2. **Export the models to ONNX** and run them with `onnxruntime`, which is much smaller than PyTorch.
3. **Quantise to 8-bit** (int8), which makes each model about four times smaller, usually at a small cost in accuracy that we'd measure first.

## What I learned

- **Deploying is a measurement problem.** The useful numbers weren't the model's accuracy but its memory, start-up time and package size, and those decided which hosts were even possible.
- **Ship the smallest thing that behaves the same.** Thumbnails cut the package by three-quarters with no change in what visitors see, because we checked what the app really reads (embeddings, not photos).
- **Pin everything.** The same versions as `uv.lock`, the same file paths, the same `app.py`: the online copy should differ from the laptop only in where it runs.
- **Free hosting is shrinking.** Hugging Face Spaces used to host Gradio apps like this one for free. Today a model app needs a paid plan or a card on file, so it's worth knowing your app's memory before you pick a host.

## Files

| File | What it does |
|---|---|
| `scripts/stage9_deploy.py` | builds the package, measures it, writes `reports/stage9_report.json`; `--upload` sends it to a Space |
| `dogbreeds/stages/stage9.py` | the "Stage 9" page |
| `reports/stage9_report.json` | the sizes and measurements |
| `tests/test_deploy.py` | every photo the app shows is packed, versions match `uv.lock`, the Space settings, the port |
| `app.py` | reads `GRADIO_SERVER_PORT`, so the same file runs locally and on a server |
