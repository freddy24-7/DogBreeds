# Stage 6: fine-tuning the backbone

**Goal:** stage 5 showed that a new head on *frozen* features can't beat ImageNet's head, because both read the same 1,408 numbers. So now we train the **last two blocks of the backbone** too, on real photos with **data augmentation**, so the features themselves can adapt to dog breeds. Then we do an **error analysis** of the look-alike pairs.

**Result:** the "Stage 6" tab. It shows ImageNet's head and the fine-tuned model side by side on any photo, a three-way comparison (stages 4, 5 and 6), the look-alike pair analysis, and the training curves.

**The outcome, up front:** fine-tuning made the model **slightly worse**. Top-1 went from 95.1% to 94.2% on Stanford Dogs and from 90.3% to 89.6% on the Oxford check. The ship rule, written down **before** training, isn't met. **Verdict: we don't ship it.** Together with stage 5, this teaches the most useful lesson of the project so far: *when is fine-tuning worth it?*

## What changes compared with stage 5

```
stage 5:  photo -> blocks 0-8 (all frozen) ------------------------------> our head (trained)
stage 6:  photo -> blocks 0-6 (frozen) -> blocks 7-8 (TRAINED) -> avgpool -> our head (trained)
```

| | Stage 5 | Stage 6 |
|---|---|---|
| What's trained | the head (0.17M weights) | blocks 7–8 + the head (**3.4M weights**) |
| What it trains on | cached features | **real photos**, every epoch |
| Data augmentation | impossible (features are fixed) | random crop, flip and colour, every epoch |
| Time per training run | 25 seconds | **30 minutes** |
| Model file | 0.7 MB | 13.7 MB (only the tuned blocks + head; blocks 0–6 are plain ImageNet) |

## Step by step

All steps are in `scripts/stage6_finetune.py`. The building blocks are in `dogbreeds/finetune.py`, under "stage 6".

### Step 0: decide the ship rule before training

Before seeing any result, we wrote down what "better" means (`MIN_OXFORD_GAIN` in the script):

1. **At least +1 point top-1 on the Oxford check.** These photos are unseen, and 1 point is more than the noise (±0.65 points on 2,086 photos).
2. **At least as many 🟢 answers in the app** as ImageNet's head gives (87.4% of dogs, with stage 3's rules).

**Why before?** After an hour of training, it's tempting to find *some* number that improved and call it a success. A rule fixed in advance prevents that. It's the same idea as a test written before the code.

### Step 1: the same split as stage 5

The same 10,800 training and 1,200 validation photos (10 per breed), with the same seed, so stages 5 and 6 are directly comparable.

### Step 2: build the model

- A **fresh copy** of EfficientNet-B2 with ImageNet weights. It isn't the app's shared model, which must stay unchanged.
- **Blocks 0–6 frozen** (`requires_grad = False`, 4.5M weights). Early blocks detect general things such as edges, textures and shapes, which are useful for any photo.
- **Blocks 7–8 trainable** (3.2M weights). The last blocks combine those into high-level features such as "pointy ears" or "curly coat". These are the most task-specific, so they're the ones worth adapting.
- **The head starts as ImageNet's head** (the 120 dog rows), stage 5's best start. So at epoch 0 the model is exactly the stage 4 baseline. A test checks this.
- **BatchNorm stays in eval mode** (`PartlyFrozen.train()`). BatchNorm layers normalise using statistics. In training mode they re-estimate those from each batch, and 32 photos is too few to estimate them well, so we keep ImageNet's statistics. Only the head's dropout switches on during training.

### Step 3: train with data augmentation

Every epoch, each training photo is changed at random before the model sees it (`finetune.train_transform`):

| Change | Why |
|---|---|
| Random crop of 35–100% of the photo, resized to 288×288 | Dogs appear at any size and position |
| Random horizontal flip | A dog facing left is the same breed |
| Small random changes in brightness, contrast and colour | Different light and cameras |

The model never sees exactly the same picture twice, which makes memorising harder. The validation photos are *not* augmented: they're prepared exactly as in the app.

The training loop is the same as stage 5's: forward, loss, `zero_grad`, `backward`, `step`. Two things are new:

- `loss.backward()` now computes gradients through blocks 7–8 as well. The frozen blocks run under `torch.no_grad()`, so no time or memory is spent on gradients they don't need.
- Learning rate 0.0001, ten times smaller than stage 5's head. Pre-trained weights are already good and need small nudges. A big step would wreck what they learned.

Six epochs took **30 minutes** on the M1 GPU, with one data-loading process (the lesson from stage 4 about the 8 GB Mac). The best weights are saved after every epoch, so a crash wouldn't lose the work.

**What happened:**

| Epoch | 0 (= stage 4) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Validation accuracy | **94.7%** | 93.6% | 93.3% | 92.8% | 92.7% | 92.0% | 92.2% |
| Validation loss | 0.197 | **0.175** | 0.190 | 0.197 | 0.201 | 0.234 | 0.238 |
| Training loss | – | 0.278 | 0.200 | 0.173 | 0.145 | 0.120 | 0.104 |

Validation accuracy **fell from the very first epoch**, while training loss kept falling. See [docs/stage6_training.png](docs/stage6_training.png).

### Step 4: test once, next to stages 4 and 5

| | ImageNet's head (stage 4) | Our head (stage 5) | Fine-tuned (stage 6) |
|---|---|---|---|
| Stanford Dogs top-1 | **95.1%** | 95.0% | 94.2% |
| **Oxford check top-1** | **90.3%** | 90.1% | 89.6% |
| Calibration error, Oxford (ECE) | 0.039 | **0.014** | 0.026 |
| Dogs that get a 🟢 name (app) | **87.4%** | 79.8% | 81.0% |

On the 2,086 unseen Oxford photos, fine-tuning **fixes 30** answers and **breaks 44**.

### Step 5: error analysis of the look-alike pairs

This was the hope: the look-alike pairs from stage 4 differ in details (coat, ears, size) that frozen features might not capture but tuned features could. To test it, "both right" counts all test photos of **both** breeds together. Moving the border between them (stage 5's effect) can't improve this number; only telling them apart better can.

| Pair (A / B) | Both right: stage 4 → 5 → 6 | A called B · B called A: stage 4 → 6 |
|---|---|---|
| Eskimo dog / Siberian husky | 69% → 69% → 69% | 18 · 12 → 9 · 27 |
| collie / Border collie | 84% → 84% → 77% | 13 · 0 → 6 · 9 |
| American Staffordshire terrier / Staffordshire bullterrier | 82% → 84% → 82% | 15 · 6 → 8 · 8 |
| English foxhound / Walker hound | 84% → 86% → 79% | 12 · 1 → 10 · 2 |
| miniature poodle / toy poodle | 86% → 87% → 82% | 8 · 1 → 6 · 7 |
| Siberian husky / malamute | 86% → 78% → 78% | 5 · 6 → 2 · 3 |

**No pair improved.** Eskimo dog vs husky is the clearest case: 69% both right, three times. The mistakes just swapped direction (18 Eskimo dogs called huskies became 9, but 12 huskies called Eskimo dogs became 27). As in stage 5, the border moved and the separation didn't improve. See also [docs/stage6_breed_changes.png](docs/stage6_breed_changes.png): Border collie fell from 100% to 80%.

### Step 6: thresholds and the ship rule

Its own stage 3 thresholds are a dog threshold of 0.42 (unchanged, from ImageNet's branch) and a breed threshold of 0.85. With them, 81.0% of dogs get a 🟢 name, at 96.0% accuracy.

| Ship rule | Needed | Got | |
|---|---|---|---|
| Oxford top-1 gain | ≥ +1.0 point | −0.7 points | ❌ |
| Dogs with a 🟢 name | ≥ 87.4% | 81.0% | ❌ |

**Don't ship.**

### Step 7: save

`models/stage6_b2.pth` (not in git), `reports/stage6_report.json`, `config/stage6_thresholds.json` and `docs/stage6_*.png`.

In the app, `FineTuned6Predictor` runs blocks 0–6 **once** and then branches: ImageNet's blocks 7–8 and head give the dog share (stage 3's "is it a dog?"), and our tuned blocks 7–8 and head give the breed. Only the last two blocks run twice.

## Why fine-tuning didn't help here

1. **The model has probably already trained on these exact photos.** Stanford Dogs was made from ImageNet. The pre-trained model was trained on about 1,300 photos per breed, very likely including our 90 training photos per breed. On this task, with this kind of photo, it's already close to the best it can be. Training it further on a **smaller sample of the same data** can only pull it away from what it learned on the larger set.
2. **Too little data for 3.4 million weights.** 10,800 photos are enough to nudge a head (stage 5), but blocks 7–8 have 20 times more weights. They start fitting the details of these 10,800 photos, as the falling training loss and rising validation loss show. A real but mild case of **forgetting**: the model loses some of what it knew while learning the small dataset.
3. **The unseen photos confirm it.** If it were only leakage flattering the baseline on Stanford, the Oxford photos would have shown a gain. They didn't: −0.7 points there too.

**When fine-tuning does pay off** is when your data is **different** from what the model was trained on:
- new classes (breeds ImageNet doesn't have, like the four Oxford breeds that Stanford lacks);
- a different kind of photo (phone snapshots at a dog park, night-time camera traps, X-rays);
- a different task (detecting where the dog is, rather than which breed).

Our setup is the opposite: the same classes, the same kind of photos, and less of them. That's why it didn't help.

**What would likely help instead** (not done here): a **bigger or newer pre-trained model** (for example a larger EfficientNet, a ConvNeXt or a vision transformer), **test-time augmentation** (average the predictions over a few crops and flips), or **more and different data** such as photos taken by users.

## A detail worth noticing: loss and accuracy disagreed

Early stopping kept **epoch 1** because it had the lowest validation **loss** (0.175). But its validation **accuracy** (93.6%) was already lower than epoch 0's (94.7%).

The two measure different things:
- **Accuracy** only counts right or wrong.
- **Loss** also rewards being *confident when right* and punishes being *confident when wrong*. Epoch 1 became more confident (better calibrated), which lowered the loss even though it got a few more photos wrong.

So the choice of stopping metric matters. Stopping on accuracy would have kept epoch 0, which is the unchanged ImageNet model: an honest "fine-tuning didn't help" from the validation set alone.

## Verdict

**Don't ship it.** The app keeps ImageNet's head, as stage 5 also concluded.

Stages 5 and 6 together tell one story. With a strong pre-trained model that already knows your classes from much more data, **the pre-trained model is the right choice**, and a good evaluation (stage 4) is what lets you prove it instead of assuming that "we trained it ourselves" means better. In an interview: "I tried a new head and then partial fine-tuning; both lost to the pre-trained baseline on independent data, because the baseline had been trained on a superset of my data. Fine-tuning pays off for data that's different, not for less of the same."

## Concepts

- **Fine-tuning vs feature extraction:** stage 6 trains part of the backbone; stage 5 only trained a head on fixed features.
- **Which layers to unfreeze:** early layers learn general features (edges, textures) and late layers learn task-specific ones. Unfreeze from the end.
- **Data augmentation:** random changes to training photos so the model learns the breed, not the exact picture. It needs real photos, not cached features.
- **Small learning rates for pre-trained weights:** 0.0001 instead of 0.001. Nudge, don't overwrite.
- **BatchNorm in eval mode during fine-tuning:** with small batches, keep the pre-trained statistics.
- **Forgetting:** training on a small dataset can erase part of what the model learned on a large one.
- **A ship rule decided in advance:** protects you from finding a metric that happens to look good after the fact.
- **Error analysis per pair:** "both right" separates *telling breeds apart* from *moving the border between them*.
- **Loss vs accuracy:** they can disagree. Choose the metric you stop on deliberately.

## Code

| File | What's in it |
|---|---|
| `dogbreeds/finetune.py` | New for stage 6: `train_transform()` (augmentation), `PartlyFrozen` (frozen + tuned blocks + head; BatchNorm stays in eval mode), `train_partly_frozen()` (the training loop on photos, with a checkpoint every epoch), `predict_photos()`. |
| `scripts/stage6_finetune.py` | New. Steps 0–7 above. The ship rule is at the top. |
| `dogbreeds/predictors.py` | `FineTuned6Predictor`: shares blocks 0–6, then branches. A subclass of stage 5's predictor; only `_logits()` differs. |
| `dogbreeds/stages/stage6.py` | New. The tab. |
| `tests/test_finetune.py` | New tests: augmentation gives a different picture each time; at epoch 0 the model equals ImageNet's; training changes only the unfrozen blocks (and BatchNorm stays in eval mode); the stage 6 predictor keeps ImageNet's dog check; a valid report. |

**Rerun:**

```bash
uv run python scripts/stage6_finetune.py              # reuses the saved model and predictions: seconds
uv run python scripts/stage6_finetune.py --retrain    # trains again: about 40 minutes
```

To try other settings, change `FIRST_TRAINED_BLOCK`, `EPOCHS` or `LR` at the top of the script and run with `--retrain`. Keep the ship rule as it is: it was set before the first run.
