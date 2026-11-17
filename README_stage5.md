# Stage 5: our own head (transfer learning)

**Goal:** train something ourselves for the first time. We keep the pre-trained model's "eyes", replace its "brain" with a new one for our 120 breeds, and train only that. Then we check honestly whether it beats the stage 4 baseline.

**Result:** the "Stage 5" tab. Upload a photo and see **both heads side by side** (ImageNet's head and ours), each with its own stage 3 thresholds. Below that are the comparison with the baseline, the per-breed changes, and the training curves.

**The outcome, up front:** our head is **not more accurate** (95.0% vs 95.1%, and 90.1% vs 90.3% on the Oxford check). It is **much better calibrated**, but in the app it gives fewer 🟢 answers. **Verdict: we don't ship it.** The app keeps ImageNet's head. That's a normal and useful result, and this README explains why it happened.

In software terms: we built a new version of a component, ran it through the same regression suite (stage 4), and the suite said "no improvement". Without stage 4, we might have shipped it because "we trained it ourselves, so it must be better".

## The idea: backbone and head

An ImageNet model has two parts:

```
photo ──> [ backbone ] ──> 1,408 numbers ──> [ head ] ──> 1,000 scores (one per ImageNet class)
           almost all       ("features":       one Linear
           of the model      fur, ears,        layer
           (7.7M weights)    snout, ...)       (1.4M weights)
```

- **The backbone** turns a photo into 1,408 numbers that describe what's in it. ImageNet trained it on 1.28 million photos, and it's good at this. We **freeze** it: its weights never change.
- **The head** is a single `Linear(1408, 1000)` layer: a score for each class is a weighted sum of the 1,408 features. We **replace it** with a new `Linear(1408, 120)`, one output per Stanford Dogs breed, and train only that.

This is **transfer learning**: reuse what a big model learned, and train only a small part for your own problem. It's the most common way deep learning is used in practice.

## Step by step

All steps are in `scripts/stage5_train_head.py`. The building blocks are in `dogbreeds/finetune.py`.

### Step 1: cache the features (the slow part, done once)

Because the backbone is frozen, the 1,408 numbers for a photo never change. So instead of running the backbone again in every training epoch, we run it **once** per photo and save the result:

| Photo set | Photos | Used for |
|---|---|---|
| Stanford Dogs train | 12,000 | training and validation |
| Stanford Dogs test | 8,580 | the stage 4 test (final report only) |
| Oxford Pets + food, "trainval" | 3,905 | choosing stage 3 thresholds |
| Oxford Pets + food, "test" | 3,744 | the Oxford check and testing the thresholds |

That's 28,229 photos, which took **about 9 minutes** on the M1 GPU. The result is 28,229 × 1,408 numbers, about 160 MB in `data/stage5_features_b2_*.npz`.

How the backbone is cut off (`finetune.backbone`): torchvision's EfficientNet runs `features → avgpool → flatten → classifier`. We run everything except `classifier`, which is the head.

**Why it matters:** training on cached features takes **seconds** instead of hours, so we can try several settings. The price is that we can't use **data augmentation** (random crops and flips during training), because the features are fixed. That's one of the things stage 6 adds.

### Step 2: check the cache

Mistakes in this kind of pipeline are silent: a wrong transform or a wrong layer still produces numbers. So before training, we put ImageNet's own head on top of the cached features and compare with stage 4's predictions:

- the top-1 breed is the same for **100%** of the 8,580 test photos;
- the largest probability difference is 0.000003 (rounding on the GPU);
- top-1 accuracy is **95.1%**, exactly as in stage 4.

The script stops if this check fails.

### Step 3: split train into train and validation

| Set | Photos | Job |
|---|---|---|
| **Training** | 10,800 (90 per breed) | The head learns from these |
| **Validation** | 1,200 (10 per breed) | Compare settings and decide when to stop |
| **Test** | 8,580 (Stanford) + 2,086 (Oxford) | Final report, **looked at once**, after all choices are made |

The split is **stratified**: the same 10% is taken from every breed, so no breed is missing from validation. It uses a fixed seed, so it's the same split every run.

Why a separate validation set: if we picked the best setting by looking at the test results, the test score would be flattered, because we'd have tuned on it. It's the same idea as stage 3 choosing thresholds on "trainval" and reporting on "test".

### Step 4: train the head

This is the classic PyTorch training loop from the course (`finetune.train_head`). For every batch of 256 photos:

```python
logits = head(features)              # 1. forward pass: 120 scores per photo
loss = loss_fn(logits, labels)       # 2. how wrong is it? (cross-entropy)
optimizer.zero_grad()                # 3. clear the previous gradients
loss.backward()                      # 4. backpropagation: how should each weight change?
optimizer.step()                     # 5. change the weights a little (AdamW)
```

One **epoch** is one pass over all 10,800 training photos. After each epoch, we measure loss and accuracy on the validation photos.

We tried **four settings**, 30 epochs each, with two questions:

1. **Where does the head start?** A *random start* (a new head that knows nothing: 0.7% accuracy, which is guessing 1 in 120), or a copy of **ImageNet's head**, keeping only the 120 rows of our breeds. That copy gives exactly stage 4's predictions before any training (a test checks this).
2. **Learning rate:** how big each weight change is. Too small means slow learning. Too big means jumpy learning and overfitting.

| Setting | Best epoch | Validation accuracy |
|---|---|---|
| random start, learning rate 0.001 | 30 | 92.9% |
| random start, learning rate 0.01 | 6 | 92.7% |
| ImageNet start, learning rate 0.0001 | 30 | 94.3% |
| **ImageNet start, learning rate 0.001** | **4** | **94.7%** ✅ chosen |

**Early stopping:** from each run we keep the weights of the epoch with the **lowest validation loss**, not the last epoch. In the chosen run, that's epoch 4. After that, the training loss keeps falling, but the validation loss slowly rises. That's **overfitting**: the head starts memorising the training photos instead of learning breeds. You can see it in the right half of [docs/stage5_training.png](docs/stage5_training.png).

**Noise:** with 1,200 validation photos, one photo is 0.08 points, and the uncertainty on an accuracy around 95% is about ±0.6 points. So the two ImageNet-start settings are a tie, and the choice between them is close to a coin toss. That's fine: the test below shows whether it matters.

### Step 5: test once

The chosen head is tested on exactly the stage 4 photos, with the same `model_report()`:

| | ImageNet's head (stage 4 baseline) | Our head |
|---|---|---|
| Stanford Dogs top-1 | 95.1% | 95.0% |
| Stanford Dogs top-3 | 99.6% | 99.6% |
| **Oxford check top-1** (unseen photos) | **90.3%** | **90.1%** |
| Oxford check top-3 | 98.4% | 98.3% |
| Calibration error, Stanford (ECE) | 0.064 | **0.028** |
| Calibration error, Oxford (ECE) | 0.039 | **0.014** |

On the 2,086 Oxford photos, our head **fixes 24** answers and **breaks 29**.

### Step 6: our head's own stage 3 thresholds

A new model has its own confidence, so it needs its **own thresholds**, measured with stage 3's rules on Oxford "trainval" photos and reported on "test":

- **Dog threshold: 0.42, unchanged.** "Is it a dog?" still comes from ImageNet's head (dog share), because our head only knows dog breeds. It would give a cat a breed too.
- **Breed threshold: 0.82** (was 0.64). That's what our head needs for 🟢 answers to be at least 95% right.

| On Oxford test photos | ImageNet's head | Our head |
|---|---|---|
| Dogs that get a 🟢 breed name | **87.4%** | 79.8% |
| Accuracy of the 🟢 names | 95.4% | 96.5% |

### Step 7: save

- `models/stage5_head_b2.pth`: the head's weights (680 KB) plus the breed order. It's not in git; rerun the script to make it.
- `reports/stage5_report.json`: all numbers, including every epoch of every setting.
- `config/stage5_thresholds.json`: step 6's thresholds.
- `docs/stage5_training.png`, `docs/stage5_breed_changes.png` and `docs/stage5_thresholds.png`.

In the app, `FineTunedPredictor` (in `dogbreeds/predictors.py`) runs the backbone **once** per photo and puts **both heads** on top. ImageNet's head gives the dog share and our head gives the breed, so the side-by-side comparison costs hardly any extra time.

## Finding 1: no gain in accuracy, and why that makes sense

- Our head learns from **90 photos per breed**. ImageNet's head learned from about **1,300 photos per breed**, and Stanford Dogs' photos come from ImageNet, so it has probably seen these same training photos already.
- Both heads use the **same frozen features**. A new head can only re-weigh the same 1,408 numbers. If two breeds look the same in those numbers, no head can separate them.
- The random-start heads show this too. With only our 90 photos per breed they reach about 93%, below ImageNet's head. Starting from ImageNet's head and training a little hardly moves it.

**What this means:** a linear head on frozen features only helps when your classes are **new to the model**, for example breeds ImageNet doesn't have, or photos that look very different, like X-rays. Here, ImageNet had already done this exact job, with far more data. To do better, the **backbone itself** has to learn: that's stage 6.

## Finding 2: the head moved the border between look-alikes

See [docs/stage5_breed_changes.png](docs/stage5_breed_changes.png). The biggest changes come in **pairs**:

| Breed that gained | | Its look-alike that lost | |
|---|---|---|---|
| Eskimo dog | 46% → 66% | Siberian husky | 82% → 71% |
| English foxhound | 74% → 84% | Walker hound | 94% → 89% |
| American Staffordshire terrier | 75% → 84% | Staffordshire bullterrier | 89% → 84% |
| miniature poodle | 78% → 85% | toy poodle | 94% → 88% |

Our head didn't learn to tell these pairs apart **better**. It moved the **border** between them: more photos are now called Eskimo dog, so more huskies are wrongly called Eskimo dog as well. The overall accuracy barely changes. This is why you always look **per class**: the average hides a shift like this.

## Finding 3: better calibrated, but fewer 🟢 answers

- **Calibration is much better** (ECE 0.039 → 0.014 on Oxford). In stage 4 the model was *underconfident*: it said 65% and was right 81% of the time. Our head says 65% and is right about 73% of the time. The most likely reason is that our plain training (cross-entropy without label smoothing) dropped the caution that was built into ImageNet's own training.
- **But calibration isn't what the app needs.** The app needs confidence that **separates right answers from wrong ones**, so that a threshold can keep the wrong ones out of 🟢. That's called **discrimination** or ranking, and it's a different property from calibration. Our head is a little worse at it. To keep 🟢 answers 95% right, it needs a threshold of 0.82, and then only 80% of dogs get a name instead of 87%.
- **The lesson:** judge a model by the metric the **product** uses, here "how many dogs get a reliable 🟢 answer", not only by the metric that improved.

## Verdict

**Don't ship it.** The app (stage 3) keeps ImageNet's head. A model that is no more accurate and answers 🟢 less often is not an improvement, however good its calibration looks.

That isn't a failed stage. Teams usually go through exactly this: train, compare with the baseline, and decide **not** to add complexity. It's also something good to tell in an interview: "I trained a head, it didn't beat the pre-trained baseline on independent data, and here's why."

**Stage 6** goes further: unfreeze the **last layers of the backbone** so the features themselves can adapt, add data augmentation (which needs real photos instead of cached features), and look at the look-alike pairs from Finding 2.

## Concepts

- **Transfer learning:** reuse a pre-trained model and train only a part of it for your own problem.
- **Backbone and head, freezing:** a frozen part's weights don't change in training (`requires_grad=False`, or, as here, it isn't given to the optimiser at all).
- **Feature caching:** run the frozen part once and train on its saved output. It's fast, but it means no data augmentation.
- **Train / validation / test:** learn on train, choose settings on validation, report once on test.
- **The training loop:** forward, loss, `zero_grad`, `backward`, `step`, and epochs and batches.
- **Learning rate:** the step size. 0.01 learned fastest but overfitted earliest (best at epoch 6, then validation loss rose).
- **Overfitting and early stopping:** training loss keeps falling while validation loss rises, so keep the best epoch.
- **Initialisation:** starting from a good head (ImageNet's) beats starting from random when your data is small.
- **Noise in measurements:** ±0.6 points on 1,200 photos. Differences smaller than that are ties.
- **Calibration vs discrimination:** "is 80% really 80%?" vs "does higher confidence mean more likely right?". The app's thresholds need the second.

## Code

| File | What's in it |
|---|---|
| `dogbreeds/finetune.py` | New. `backbone()`, `extract_features()`, `BreedHead`, `head_from_imagenet()`, `stratified_split()`, `train_head()` (the training loop), and saving and loading. |
| `scripts/stage5_train_head.py` | New. Steps 1–7 above. |
| `dogbreeds/predictors.py` | `FineTunedPredictor`: one backbone pass and two heads. It's added to `PREDICTORS` once the head file exists, so the predictor contract tests run on it automatically. |
| `dogbreeds/decision.py` | `load_thresholds()` can read stage 5's own threshold file. |
| `dogbreeds/stages/stage5.py` | New. The tab: side-by-side comparison, results, training. |
| `tests/test_finetune.py` | New. Backbone + head = the whole model; the ImageNet start equals stage 4; the stratified split; training learns an easy problem; both heads share one pass; a valid report. |

**Rerun:**

```bash
uv run python scripts/stage5_train_head.py
```

The first run takes about 10 minutes (step 1). After that, about 25 seconds, because the features are reused. To try other settings, edit `EXPERIMENTS` or `EPOCHS` at the top of the script and rerun. Delete `data/stage5_features_b2_*.npz` to recompute the features.
