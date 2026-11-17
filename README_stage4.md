# Stage 4: the model report

**Goal:** measure the model properly, once, on a fixed test set, and save the result as the **baseline** every later model has to beat.

**Result:** the "Stage 4" tab, a report rather than an upload page. It shows accuracy over 120 breeds, a leakage check, calibration, the hardest breeds, and the most-confused pairs with example photos.

In software terms: stage 3 was input validation and error handling. **Stage 4 is the benchmark and the regression test suite.**

## The test sets

| Test set | Photos | Breeds | Why |
|---|---|---|---|
| **Stanford Dogs test** | 8,580 | 120 | The main benchmark: official split, all breeds |
| **Oxford-IIIT Pets test, dogs** | 2,086 | 21 | The leakage check: the same breeds, different photos |
| **Your own photos** (optional) | – | any of the 120 | The most honest test. See "Add your own photos" below |

**Stanford Dogs' 120 breeds are not exactly ImageNet's 118.** They're 117 of ImageNet's dog breeds (only the **Dalmatian** is missing) plus **3 wild dogs**: dingo, dhole and African hunting dog. Each breed is matched to its ImageNet class by name. There's one trap: ImageNet has two classes called "Cardigan", the corgi (264) and a **cardigan sweater** (474). A test makes sure we picked the dog.

**How a prediction is made.** As in stage 3: softmax over all 1000 ImageNet classes, keep the 120 breed classes, and rescale them to add up to 1. That gives "if it's a dog, which breed?".

## Results

| | EfficientNet-B0 | EfficientNet-B2 |
|---|---|---|
| **Stanford Dogs, top-1** | 93.0% | **95.1%** |
| Stanford Dogs, top-3 | 99.2% | 99.6% |
| Stanford Dogs, top-5 | 99.8% | 99.9% |
| Calibration error (ECE, lower is better) | 0.089 | 0.064 |
| *Leakage check, the same 21 breeds:* | | |
| Stanford Dogs, top-1 | 94.8% | 97.1% |
| Oxford Pets, top-1 | 89.2% | 90.3% |
| **Gap** | **5.6 points** | **6.8 points** |
| Speed, batches of 32 on the M1 GPU | ~10 ms per photo | ~21 ms per photo |

All exact numbers are in [reports/stage4_report.json](reports/stage4_report.json).

## Finding 1: the benchmark flatters the model (leakage)

- On the same 21 breeds, B2 scores **97.1%** on Stanford Dogs but **90.3%** on Oxford Pets.
- **Stanford Dogs was made from ImageNet photos**, so the pre-trained model has probably seen many of these "test" photos during its own training. It's like a student who has seen the exam questions.
- We can't prove it without ImageNet itself, and part of the gap may come from the photos (Oxford's are more everyday snapshots). But it's a consistent gap of 6–7 points, for both models.
- **What this means:** the realistic accuracy for users is **about 90%**, not 95%. The Stanford number is still useful for *comparing* models, as long as they're all compared on the same test set. For stage 5 we need to keep this in mind. A fine-tuned model trained on Stanford Dogs' *train* photos can still be compared on its *test* photos, but the Oxford check stays the more honest one.

**In an interview:** "my test set overlaps with the pre-training data" is a common real-world mistake (also with LLMs and benchmarks). Checking it against an independent dataset, as here, is the standard fix.

## Finding 2: the mistakes are between look-alike breeds

The most-confused pairs for B2 (true breed → taken for):

| True breed | Taken for | Share of that breed's photos |
|---|---|---|
| Eskimo dog | Siberian husky | 36% |
| collie | Border collie | 25% |
| American Staffordshire terrier | Staffordshire bullterrier | 23% |
| English foxhound | Walker hound | 21% |
| miniature poodle | toy poodle | 15% |
| Siberian husky | Eskimo dog | 13% |

- These are pairs that even experts confuse, and some differ mainly in size, which a photo doesn't show (toy vs miniature poodle).
- Eskimo dog is the hardest breed by far (46% right). The model mostly calls it a husky.
- 23 breeds are 100% right, among them the komondor (a dog that looks like a mop) and the Leonberger.
- **The model's mistakes aren't random.** They follow real ambiguity. The 🟡 "not sure" answers from stage 3 are exactly these cases, and showing the top 3 nearly always includes the right breed (top-3: 99.6%).

See [docs/stage4_hardest_b2.png](docs/stage4_hardest_b2.png) and the example photos in the app.

## Finding 3: the model is *under*confident

**Calibration** asks: when the model says "80% sure", is it right 80% of the time? See [docs/stage4_calibration.png](docs/stage4_calibration.png).

- Both curves lie **above** the diagonal. When B2 says about 65%, it's right about **81%** of the time. When it says about 45%, it's right about 56%.
- So the model is **more often right than it claims**. That's the opposite of the usual worry (overconfidence).
- Two likely reasons:
  1. **Leakage again:** on photos it has seen, it does better than its probabilities suggest. On Oxford Pets, B2's calibration error is lower (0.039 vs 0.064).
  2. **Look-alike breeds share the probability.** "Husky 55%, Eskimo dog 40%" is a low top probability, but the top choice is still often right.
- **What this means for the app:** stage 3's breed threshold (0.64) was measured on Oxford photos, so it's already set on realistic data. If we used Stanford's numbers to choose it, we'd set it too low. A fix for calibration itself is **temperature scaling** (one number that sharpens or softens the probabilities), which is a possible later addition.

**ECE** (expected calibration error) is the average distance between confidence and accuracy, weighted by the number of photos at each confidence level. 0 is perfect.

## Finding 4: B2 vs B0

- B2 is **about 2 points more accurate** and **better calibrated**, at about twice the compute.
- Both are fast. Even on the CPU, one photo takes 150–230 ms (stage 2). In batches on the M1 GPU it's 10–21 ms per photo. That's the usual effect of **batching**: the GPU processes 32 photos almost as fast as one.
- B2 stays the default.

## Concepts

- **A fixed test set as a baseline.** Same photos, same metrics, saved to a file. A model change counts as an improvement only if it beats this report. It's the ML version of a regression test.
- **Top-1, top-3, top-5 accuracy.** Top-3 matters here, because the app shows 3 breeds.
- **Per-class accuracy.** An average of 95% hides a breed at 46%. Always look per class.
- **Confusion matrix and confused pairs.** `cm[true, predicted]` counts photos. Its largest values off the diagonal are the most common mistakes.
- **Calibration and ECE.** Whether confidence can be trusted.
- **Leakage / contamination.** Test data the model has seen before makes results look better than reality.
- **Datasets and DataLoaders as reusable code.** `dogbreeds/data.py` gives every dataset in the same shape (image, breed number 0–119), so the same evaluation code (and in stage 5, training code) works on all of them.

## Add your own photos (optional, the most honest test)

Put photos in folders named after the breed, then rerun the report:

```
data/my_photos/golden retriever/IMG_0012.jpg
data/my_photos/Labrador retriever/any_name.jpg
data/my_photos/Border collie/...
```

- Folder names must be one of the 120 breeds. Spelling is matched loosely, so `golden_retriever` works too, and unknown folders are listed and skipped.
- iPhone HEIC photos: export them as JPEG first.
- Then run:

```bash
uv run python scripts/stage4_report.py
```

Your photos appear as an extra row in the report. Only the new photos are processed; the earlier predictions are reused.

## Code

| File | What's in it |
|---|---|
| `dogbreeds/data.py` | New. `stanford_breeds()` (matching to ImageNet), `PhotoDataset`, and loaders for Stanford Dogs, the Oxford dogs and your own photos. Reused for training in stage 5. |
| `dogbreeds/evaluate.py` | New. Batch predictions and the metrics: top-k, per breed, confusion pairs, calibration and ECE, all in `model_report()`. |
| `scripts/stage4_report.py` | New. Runs both models on all test sets, writes `reports/stage4_report.json` and the plots in `docs/`. Saves raw predictions in `data/`, so reruns take seconds. |
| `dogbreeds/stages/stage4.py` | New. The report tab. |
| `tests/test_evaluate.py` | New. Every metric checked on tiny examples with known answers, the breed matching (Cardigan!), and the report file. |

**Rerun:** `uv run python scripts/stage4_report.py`. It takes about 25 minutes the first time and seconds afterwards; delete `data/stage4_preds_*.npz` to really rerun the models.

**Lesson learned on an 8 GB Mac:** the first run used 4 data-loading worker processes and got stuck. Each worker loads its own copy of PyTorch, and the Mac ran out of memory and swapped. One process with batches of 32 runs smoothly, in about 25 minutes. On a machine with more memory, workers would speed it up.

**Data:** Stanford Dogs (Khosla et al., 2011; Stanford University, about 780 MB) in `data/stanford-dogs/`, not in git.
