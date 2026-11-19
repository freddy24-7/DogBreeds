"""
Stage 9: put the app online, as a Hugging Face Space.

Run from the project folder:
    uv run python scripts/stage9_deploy.py                          # build deploy/space/ only
    uv run python scripts/stage9_deploy.py --upload USER/DogBreeds  # build, then upload to that Space

The steps (README_stage9.md explains each one):
  1. Collect exactly what the app needs at runtime, and nothing else:
       - the code (app.py, dogbreeds/), measured settings (config/), reports/ and docs/
       - our own model files (models/): the stage 5 head, the stage 6 blocks, the stage 8 index
       - the example photos the pages offer, copied as they are
       - the 12,000 gallery photos of stage 8, shrunk to thumbnails (the originals are 780 MB)
     Files keep their paths, so the app runs unchanged. The ImageNet weights are NOT bundled:
     torchvision downloads them on first use, as it does on a laptop.
  2. Write the Space's own README (its settings live in a block at the top) and requirements.txt,
     with the same versions as uv.lock and the small CPU-only build of PyTorch.
  3. Measure the bundle's size.
  4. Run the bundle on its own, in a fresh Python inside deploy/space/, the way a server would: how long it
     takes to start, how long each model takes per photo, and the peak memory (what decides which hosting
     plans can run it). All of it goes into reports/stage9_report.json (shown on the stage 9 page).
  5. With --upload: create a Hugging Face Space if it doesn't exist and upload the bundle.
     Needs a Hugging Face login first (uv run hf auth login) and a PRO subscription: Hugging Face only
     hosts Gradio apps for PRO accounts. README_stage9.md lists the other options.
"""
import argparse
import importlib.metadata as metadata
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "deploy" / "space"
REPORT = ROOT / "reports" / "stage9_report.json"
GITHUB = "https://github.com/freddy24-7/DogBreeds"
THUMB = 192                     # longest side of a gallery thumbnail, in pixels (tiles are shown at about 150)
GRADIO_VERSION = metadata.version("gradio")

# Everything copied as it is. Paths relative to the project folder.
COPY = ["app.py", "dogbreeds", "config", "reports", "docs", "models",
        "data/stanford-dogs/file_list.mat"]     # the list of 120 breed folders (stanford_breeds needs it)

SPACE_README = f"""---
title: DogBreeds
emoji: 🐕
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Dog breeds from a photo, built stage by stage
---

# 🐕 DogBreeds

Upload a photo of a dog and see its breed. The app was built in nine stages, and each stage is still in it:
pick one with **Choose stage**.

The code, and a README for every stage, are on GitHub: [{GITHUB.removeprefix("https://")}]({GITHUB}).
This Space is made by `scripts/stage9_deploy.py` in that repository; don't edit it here.
"""


def referenced_photos() -> set[str]:
    """Every photo under data/ that a page shows or offers as an example (relative paths)."""
    from dogbreeds.stages import stage3

    found = {str(Path(p).relative_to(ROOT)) for p in stage3.EXAMPLES}

    def walk(obj, key=""):
        if key == "photos":                    # stage 7's per-photo measurements: not shown in the app
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, key)
        elif isinstance(obj, str) and obj.startswith("data/") and obj.lower().endswith((".jpg", ".jpeg", ".png")):
            found.add(obj)

    for report in sorted((ROOT / "reports").glob("stage*_report.json")):
        walk(json.loads(report.read_text()))
    return found


def gallery_photos() -> list[str]:
    from dogbreeds.similar import Index
    return Index.load(ROOT / "models" / "stage8_index.npz").paths


def thumbnail(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB")
    im.thumbnail((THUMB, THUMB))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=82, optimize=True)


def requirements() -> str:
    v = metadata.version
    return "\n".join([
        "# Made by scripts/stage9_deploy.py, with the versions in uv.lock.",
        "# The CPU-only PyTorch build: the Space has no GPU, and it's far smaller than the default build.",
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={v('torch')}",
        f"torchvision=={v('torchvision')}",
        f"numpy=={v('numpy')}",
        f"pillow=={v('pillow')}",
        f"scipy=={v('scipy')}",
        "# gradio itself is installed by the Space, at the sdk_version in README.md",
        "",
    ])


def size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def build() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    print("Step 1: collecting files")
    for rel in COPY:
        src, dst = ROOT / rel, OUT / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    examples = sorted(referenced_photos())
    missing = [p for p in examples if not (ROOT / p).exists()]
    assert not missing, f"photos the app refers to are missing: {missing[:5]}"
    for rel in examples:
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, OUT / rel)
    gallery = gallery_photos()
    for i, rel in enumerate(gallery):
        if not (OUT / rel).exists():           # an example photo that is also in the gallery stays full size
            thumbnail(ROOT / rel, OUT / rel)
        if i % 3000 == 0:
            print(f"  thumbnails {i:>6} / {len(gallery)}", flush=True)
    original = sum((ROOT / p).stat().st_size for p in gallery) / 1e6

    print("Step 2: the Space's README and requirements.txt")
    (OUT / "README.md").write_text(SPACE_README)
    (OUT / "requirements.txt").write_text(requirements())

    print("Step 3: measuring")
    parts = {"code (app.py, dogbreeds/)": size_mb(OUT / "app.py") + size_mb(OUT / "dogbreeds"),
             "settings, reports and plots (config/, reports/, docs/)":
                 size_mb(OUT / "config") + size_mb(OUT / "reports") + size_mb(OUT / "docs"),
             "our model files (models/)": size_mb(OUT / "models"),
             f"example photos ({len(examples)})": sum(size_mb(OUT / p) for p in examples),
             f"gallery thumbnails ({len(gallery):,})": size_mb(OUT / "data" / "stanford-dogs" / "Images")
                                                       - sum(size_mb(OUT / p) for p in examples if p.startswith("data/stanford-dogs/Images")),
             }
    files = sum(1 for f in OUT.rglob("*") if f.is_file())
    report = {"bundle_megabytes": size_mb(OUT), "files": files, "parts_megabytes": parts,
              "gallery": {"photos": len(gallery), "original_megabytes": original, "thumbnail_pixels": THUMB},
              "example_photos": len(examples), "gradio": GRADIO_VERSION,
              "not_bundled": ["ImageNet weights (torchvision downloads them on first use)",
                              "the datasets (only the photos the pages show)", "scripts/ and tests/"]}
    for name, mb in parts.items():
        print(f"  {name:58s} {mb:7.1f} MB")
    print(f"  total: {report['bundle_megabytes']:.1f} MB in {files:,} files "
          f"(the gallery photos alone were {original:.0f} MB)")
    return report


# Run inside the bundle, in a fresh Python: start the app, run every model once, report time and memory.
TRY_BUNDLE = """
import json, resource, sys, time
t = time.time()
import app                                            # builds every stage page, as the server does
start = time.time() - t
from PIL import Image
from dogbreeds.stages import stage3, stage5, stage6, stage7, stage8
img = Image.open(stage3.EXAMPLES[0]).convert("RGB")
calls = {"stage 3 (ImageNet EfficientNet-B2)": lambda: stage3.analyse(img, list(stage3.MODELS)[0]),
         "stage 5 (our head)": lambda: stage5.compare(img),
         "stage 6 (fine-tuned blocks)": lambda: stage6.compare(img),
         "stage 7 (with the dog/not-dog check)": lambda: stage7.analyse(img),
         "stage 8 (similarity search)": lambda: stage8.search(img, 8)}
seconds = {}
for name, call in calls.items():
    call()                                            # the first call also loads that stage's model
    t = time.time(); call(); seconds[name] = round(time.time() - t, 2)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3   # bytes on macOS, kilobytes on Linux
print("RESULT" + json.dumps({"start_seconds": round(start, 1), "peak_memory_megabytes": round(peak_mb),
                             "seconds_per_photo": seconds}))
"""


def try_bundle() -> dict:
    """Runs the bundle on its own (a fresh Python, inside deploy/space/) and measures it."""
    import subprocess
    out = subprocess.run([sys.executable, "-c", TRY_BUNDLE], cwd=OUT, capture_output=True, text=True, check=True)
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))
    return json.loads(line.removeprefix("RESULT"))


def upload(space: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Step 5: uploading to {space} as {api.whoami()['name']}")
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=OUT,
                      commit_message="Deploy from scripts/stage9_deploy.py")
    print(f"  done: {space_url(space)}")


def space_url(space: str) -> str:
    return f"https://huggingface.co/spaces/{space}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", metavar="USER/SPACE", help="upload the bundle to this Hugging Face Space")
    args = parser.parse_args()
    old = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report = build()
    print("Step 4: running the bundle on its own")
    report["measured"] = try_bundle()
    m = report["measured"]
    print(f"  starts in {m['start_seconds']} s, peak memory {m['peak_memory_megabytes']} MB")
    for name, s in m["seconds_per_photo"].items():
        print(f"  {name:40s} {s:.2f} s per photo")
    report["space_url"] = space_url(args.upload) if args.upload else old.get("space_url")
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    shutil.copy2(REPORT, OUT / "reports" / REPORT.name)   # the stage 9 page in the Space shows it too
    print(f"Saved {REPORT.relative_to(ROOT)}")
    if args.upload:
        upload(args.upload)


if __name__ == "__main__":
    main()
