# Stage 2: a pre-trained model, no training

**Goal:** replace the dummy with a real model and get real breed predictions, without training anything ourselves.

**Result:** the "Stage 2" tab in the app. Upload a dog photo and you get its likely breeds, a "dog share" and the model's view of all 1000 ImageNet classes.

## The key idea

Someone else already did the expensive part. **ImageNet** is a collection of about 1.3 million labelled photos in **1000 classes**: tables, cars, birds, and also **118 dog breeds**. EfficientNet was trained on all of it, which took days of GPU time. torchvision gives us the result, the trained weights, as a 20–40 MB download.

Since the model already knows 118 breeds, we don't need to train anything. We keep only the dog part of its answer.

```
photo ──► transform ──► EfficientNet ──► 1000 scores ──► softmax ──► 1000 probabilities
                                                                         │
                                                       keep classes 151–268 (the dogs)
                                                                         │
                                                                         ▼
                                                              top 3 breeds + dog share
```

## Concepts, step by step

**1. The transform must match training.** The model learned from photos that were resized, cropped and normalised in one specific way. New photos must go through exactly the same steps, or the model sees something unfamiliar. The weights come with their own transform, so we use that one:

```python
x = self.weights.transforms()(image)   # B0: resize 256, crop 224. B2: resize and crop 288
```

**2. A batch of one.** Models always expect a batch of images, so one photo becomes a batch of size 1: `unsqueeze(0)` turns `[3, 224, 224]` into `[1, 3, 224, 224]`.

**3. Eval mode and inference mode.** `model.eval()` switches off training-only behaviour such as dropout. `torch.inference_mode()` skips the bookkeeping needed for training (gradients), which makes prediction faster.

**4. Logits → softmax → probabilities.** The model outputs 1000 raw scores ("logits"). Softmax turns them into 1000 probabilities that add up to 1.

**5. Keep only the dogs.** In ImageNet's list, classes 151 up to and including 268 are dog breeds, from "Chihuahua" to "Mexican hairless". Just before them is "sea lion" and just after is "timber wolf"; a test checks these boundaries. We take the top 3 of these 118.

**6. We don't rescale.** We could rescale the 118 dog probabilities so they add up to 1 again, but we deliberately don't. Their total is the **dog share**: how much of its belief the model put on "some dog".
- A clear dog photo: the dog share is close to 100%.
- A cat photo: the dog share is low, but the app *still* shows 3 breeds, because it always picks the best 3 of 118. That's the weakness stage 3 fixes ("I don't see a dog").
- A blank, plain brown image gave a dog share of 7–8%. That's close to "no idea": if the model spread its belief evenly over all 1000 classes, the dogs would get 118/1000 ≈ 12%.

**7. Top-k.** `probs.topk(3)` returns the 3 highest probabilities and their class numbers, which we turn into names.

## Two models to compare

| Model | Parameters | ImageNet accuracy (top-1) | Input size | Time per photo (M1, CPU) |
|---|---|---|---|---|
| EfficientNet-B0 | 5.3 M | 77.7% | 224 × 224 | ~150 ms |
| EfficientNet-B2 | 9.1 M | 80.6% | 288 × 288 | ~230 ms |

The usual trade-off: bigger is a bit more accurate but slower. Both are fast enough for an app. B2 is the default. The ImageNet accuracy is over all 1000 classes; how good they are on dog breeds specifically, we measure in stage 4.

## Try this in the app

1. **A clear photo of a dog:** check whether the breed is right and the dog share is high.
2. **The same photo with B0 and B2:** do they agree? Is one more confident?
3. **A mixed breed:** the top 3 often names the parent breeds.
4. **A cat, a person or a teddy bear:** look at the dog share, and at "Under the hood" to see what the model really thinks it is.
5. **A dog breed ImageNet doesn't have**, such as a Labradoodle or a Kooikerhondje: the model has to pick something it knows.

## Limitations (fixed in later stages)

- It always names 3 breeds, even when there's no dog. → **Stage 3**
- We don't know how accurate it really is on dog photos. → **Stage 4** (test set)
- Only ImageNet's 118 breeds. → **Stage 5** (fine-tuning on Stanford Dogs, 120 breeds)

## Code

| File | What changed |
|---|---|
| `dogbreeds/predictors.py` | New `ImageNetPredictor` with `analyse()` (all details) and `predict()` (the standard interface). `DOG_CLASSES = range(151, 269)`. |
| `dogbreeds/stages/stage2.py` | The stage 2 tab: model picker, top 3, dog share and timing, and "under the hood". |
| `dogbreeds/stages/` | New: one module per stage, each builds its own tab. `app.py` only puts the tabs together. |
| `tests/` | The contract tests now also run on both EfficientNets, plus checks of the dog classes and the dog share. |

The model loads on first use, not when the app starts. The first prediction takes a second or two longer.
