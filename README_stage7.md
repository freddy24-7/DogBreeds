# Stage 7: where did the model look?

**Goal:** explain individual answers with a **heatmap** of where in the photo the model found its evidence. Then **check** those heatmaps against the truth instead of just admiring them.

**Result:** the "Stage 7" tab. Upload a photo and see:
- the stage 3 answer;
- the square the model actually sees;
- a heatmap for any of its top-3 answers.

There's also a "why breed A rather than breed B?" map, and a switch for the raw map. Below the demo are the measurements on 3,662 Oxford pets with known outlines, and a flaw in the model that the heatmaps revealed.

It explains the model the app **ships**: EfficientNet-B2 with ImageNet's head (stages 2–4). Stages 5 and 6 weren't shipped.

**The outcome, up front:**
1. **The model looks at the pet.** 84% of the heat falls on the dog, which covers about 51% of the picture, and the hottest spot is on the dog in 99% of photos. That's the same for right and wrong answers: wrong answers come from look-alike breeds, not from looking at the background.
2. **The heatmaps found a bug in the model.** Its top-left corner lights up on *every* photo, even a plain grey one. It's most likely a side effect of moving the weights from TensorFlow to PyTorch. It slightly changes some answers too.

## How a heatmap is made (CAM)

The last backbone part gives **1,408 feature maps of 9 × 9 cells**: for each cell of a 9 × 9 grid laid over the photo, 1,408 numbers describing what's there. Then:

```
feature maps A [1408, 9, 9] ──average each map over the 81 cells──> 1,408 numbers ──head──> score per breed
```

The head's score for a breed is a weighted sum of those averages. A sum of averages is the average of a sum, so:

```
score(breed) = average over the 81 cells of ( Σ_c  w[breed, c] × A[c] )  + bias
               \_______________________________________________________/
                     a 9 × 9 map: each cell's share of the score
```

That 9 × 9 map is the **class activation map (CAM)**. Cells with high values pushed the score up: that's where the model saw the breed. We resize it smoothly to 288 × 288, set negative cells to 0 (evidence *against* the breed), scale it so the hottest spot is 1, and colour it orange over the photo (`dogbreeds/heatmap.py`).

**What about Grad-CAM?** Grad-CAM is the more general method, and it works for any architecture. It weighs each feature map by the **gradient** of the score, computed with `backward()`. For a model that ends in "average, then one Linear layer", like ours, that gradient is just `w[breed, c] / 81` in every cell. So Grad-CAM gives *exactly* the CAM map, divided by 81. A test proves this with autograd (`test_cam_equals_grad_cam_for_this_model`). CAM doesn't need a backward pass, so it's faster and simpler.

**A second test checks the core property:** the average of a breed's map plus the bias **equals that breed's score**. The heatmap is a decomposition of the actual answer, not a picture loosely connected to it.

### "Why A rather than B?"

Subtracting two maps (`CAM(Saint Bernard) − CAM(boxer)`) shows where the photo looks **more like A than B**: the details that separate them. For a Saint Bernard puppy against a boxer, it lights up the white blaze and forehead markings. For look-alike pairs (stage 4), this is more useful than the normal map, which mostly says "dog face" for both breeds.

### Limits to keep in mind

- **Coarse:** 9 × 9 cells, each 32 × 32 pixels of the photo. The smooth heatmap is an interpolation, not detail the model provided.
- **"Where", not "what":** a hot spot on the head doesn't tell you whether it was the ears, the eyes or the fur colour.
- **Only the centre square:** the model's transform cuts out the centre 288 × 288 square, and anything outside it is never seen. The tab shows that square.

## Step by step: checking the heatmaps

`scripts/stage7_heatmaps.py` takes about 2 minutes.

### Step 1: the truth

Oxford-IIIT Pets has an **outline** (a "trimap") for every photo: each pixel is pet, background or border. We cut it exactly like the photo (resize, centre square) and count pet and border as "pet". A test checks that photo and outline are cut the same way.

Five outlines turned out to be **empty**: the whole picture is marked "background" even though there's a dog in it. Those are annotation errors, and we skip them, leaving 3,662 test photos. That's a small lesson of its own: **evaluation data has mistakes too**.

### Step 2: three measurements

For each photo, we take the model's answer (the top dog breed for dogs, as in stage 3, and the top ImageNet class for cats) and make its heatmap. Then:

| Measurement | Question | What chance gives |
|---|---|---|
| **Heat on the pet** | What share of the heat falls inside the outline? | The pet's share of the picture (about 50%) |
| **Pointing game** | Is the single hottest spot on the pet? | Also about 50% |
| **Mostly off the pet** | Is less than half the heat on the pet? | – |

### Step 3: the first results looked wrong

The raw maps scored well on "heat on the pet" (78%) but badly on the pointing game: **52%**, no better than chance. The example pictures showed why. Almost every heatmap had a hot spot in the **top-left corner**, even over a wall or grass.

### Step 4: investigating the corner

Before "fixing" anything, we tested whether the corner reflects the photo or the model:

| Test | Result | Meaning |
|---|---|---|
| Mirror the photo left-right | The hot spot **stays top-left** | Not about the content |
| Turn it upside down | Still top-left | Not about the content |
| A plain grey image | The corner is still extreme | Present with no content at all |
| Feature maps, all 1,408 | Top-left cell about **280×** stronger than the others | Built into the model |

**Most likely cause:** torchvision's documentation says these weights were *"ported from the original paper"*, which used TensorFlow. TensorFlow pads image edges unevenly (more on the bottom and right) for some layers, while PyTorch pads evenly. So at the top-left edge the model sees a padding pattern it never saw in training. We can't prove the cause from here, but the symptoms fit, and the effect is real either way.

**The repair** (`fix_corner`): replace the top-left cell with the average of its three neighbours, in the feature maps, before making the map.

### Step 5: results

| Oxford test photos | Photos | Pet's share of the picture | Heat on the pet | Hottest spot on the pet | Raw (with the artifact) |
|---|---|---|---|---|---|
| Dog, right breed | 1,879 | 51% | **84%** | **99%** | 78% · 52% |
| Dog, wrong breed | 202 | 46% | **79%** | **98%** | 74% · 56% |
| Dog, breed not in ImageNet | 400 | 51% | **82%** | **99%** | 78% · 71% |
| Cat (top ImageNet class) | 1,181 | 55% | **81%** | **95%** | 76% · 31% |

See [docs/stage7_heat_on_pet.png](docs/stage7_heat_on_pet.png).

## Finding 1: the model looks at the pet

- Far more heat lands on the pet (79–84%) than chance would give (46–55%), and the hottest spot is on the pet 95–99% of the time.
- **Wrong answers are not "looking at the wrong thing".** Wrong and right answers score almost the same. That matches stage 4: the mistakes are look-alike breeds, and the model looks at the right dog but confuses it with a similar one.
- **When heat spills off the pet, it's usually a small dog.** For dogs covering less than 20% of the picture, 60% of the heat is on the dog, against 86% for the others. One cell of the 9 × 9 grid is wider than such a dog, so the heat spreads onto the grass around it. That's a resolution limit of the method, not the model being distracted. See the small-dog examples in the tab.
- **No shortcuts found.** A famous failure in image models is "shortcut learning": a model that recognises huskies by the snow, or cows by the grass. On these photos there's no sign of it, since less than 4% of dogs have most of their heat off the dog. (Our test data can't rule out every shortcut, only the ones these photos would show.)

## Finding 2: the heatmaps revealed a flaw in the model

- The top-left artifact is in the model, so it affects **answers**, not only explanations. That cell contributes to every class score.
- With the corner repaired, breed accuracy on the 21 Oxford breeds goes from **90.3% to 90.6%** (11 answers fixed, 5 broken, out of 2,081). That's too small a difference to be sure it's real, so **we don't change the app's model**. But it's exactly the kind of thing to note for a future model choice. For example, a model trained natively in PyTorch shouldn't have it.
- **The lesson:** explanation tools aren't only for showing users *why*. They're a **debugging tool for engineers**. Accuracy alone (stage 4) never showed this. One look at the heatmaps did.

## Concepts

- **Class activation map (CAM):** the head's weights for one class times the last feature maps, giving each region's share of the score.
- **Grad-CAM:** the same idea with gradients as weights, so it works for any CNN. It equals CAM for "average + Linear" heads.
- **Contrastive maps:** CAM(A) − CAM(B), "why A rather than B".
- **Evaluating explanations:** don't trust a heatmap because it looks plausible. Measure it against ground truth (outlines), with metrics such as energy inside the object and the pointing game, and compare with chance.
- **Sanity checks:** mirror the input and see whether the explanation moves with the content. Feed a blank image and see whether anything lights up.
- **Shortcut learning:** a model using the background instead of the object. Heatmaps are a standard way to look for it.
- **Framework porting artifacts:** the same weights can behave slightly differently in another framework (here, padding).
- **Resolution limits:** a 9 × 9 map can't show detail smaller than about 32 pixels.

## Code

| File | What's in it |
|---|---|
| `dogbreeds/heatmap.py` | New. `model_view()` (the centre square the model sees), `mask_view()` (the outline, cut the same way), `feature_maps()`, `class_maps()` (CAM), `fix_corner()`, `to_heat()`, `overlay()`, and the two measurements. |
| `scripts/stage7_heatmaps.py` | New. Steps 1–5: runs every Oxford test photo, measures raw and repaired maps, and saves examples. |
| `dogbreeds/stages/stage7.py` | New. The tab. The model runs once per photo (`gr.State` keeps the feature maps), so switching answer, map type or raw/repaired is instant. |
| `tests/test_heatmap.py` | New. The map's average + bias = the score; CAM = Grad-CAM (with autograd); different breeds give different maps; `fix_corner` changes only one cell; the corner artifact appears on a grey image; outline and photo are cut the same way; the metrics on toy examples. |

**Rerun:**

```bash
uv run python scripts/stage7_heatmaps.py
```

It takes about 2 minutes and needs `data/oxford-iiit-pet` with `annotations/trimaps` (downloaded in stage 3).
