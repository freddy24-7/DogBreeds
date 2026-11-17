# Stage 3: knowing when it doesn't know

**Goal:** stop the app from always naming 3 breeds. It should say so when there's no dog, or when it isn't sure which breed.

**Result:** the "Stage 3" tab. Every photo gets one of three answers:

| The model sees… | The app says |
|---|---|
| Clearly a dog, clearly one breed | 🟢 **beagle** (87% sure) |
| A dog, but unsure which breed | 🟡 **A dog, but I'm not sure which breed.** Maybe a beagle, harrier or English foxhound. |
| No dog | 🔴 **I don't see a dog.** This looks most like: *tabby cat* (70%). |

## The key idea

**A model never refuses by itself.** It always gives a probability for each class it knows, so it always has a "best" breed, even for a pizza. Saying "I don't know" is something *we* add on top, with thresholds.

The important part is that we **measure** the thresholds instead of guessing them.

## Two signals, two thresholds

**1. Dog share → is there a dog?** From stage 2: the total probability on the 118 dog breeds.

```
dog share < dog threshold  →  🔴 no dog
```

**2. Breed confidence → which breed?** The top breed's share *of the dog part*:

```
breed confidence = P(top breed) / dog share
```

Example: dog share 0.80 and beagle 0.60 → beagle has 0.60 / 0.80 = **75%** of the dog part. This is a *conditional probability*: "if it's a dog, how likely is it a beagle?". The breed bars in the stage 3 tab show these values, so they now add up to at most 100% over the dog breeds.

```
breed confidence < breed threshold  →  🟡 a dog, unsure
otherwise                           →  🟢 name the breed
```

## How the thresholds were measured

**The photos:**
- **Oxford-IIIT Pets** (University of Oxford): about 7,400 photos of 25 dog breeds and 12 cat breeds. Cats are the hard case: furry, four legs, ears.
- **pizza/steak/sushi** from the course: the easy case.

**Two separate groups of photos, the same idea as a train and test split:**
- **Choose** the thresholds on the *trainval* half (3,905 photos).
- **Report** how well they work on the *test* half (3,744 photos), which played no part in choosing.
- If we chose and reported on the same photos, the result would look better than it really is.

**The rules for choosing.** These are decisions, not facts. They live at the top of [scripts/stage3_thresholds.py](scripts/stage3_thresholds.py):
- **Dog threshold:** wrongly reject at most **2%** of real dogs. Turning away someone's actual dog is the most annoying mistake for a user.
- **Breed threshold:** when the app names a breed (🟢), it must be right at least **95%** of the time. Below that, it says 🟡 and shows the top 3.

For breed accuracy we can only use the 21 Oxford dog breeds that ImageNet also has. The other 4 (American bulldog, American pit bull terrier, Havanese, Shiba Inu) still count as dogs for the dog threshold.

## Results on the test photos

| | EfficientNet-B0 | EfficientNet-B2 |
|---|---|---|
| **Dog threshold** (minimum dog share) | 0.48 | 0.42 |
| **Breed threshold** (minimum breed confidence) | 0.69 | 0.64 |
| Dogs correctly accepted | 97.8% | 97.5% |
| Cats correctly rejected | 94.5% | 95.7% |
| Food correctly rejected | 100% | 100% |
| Dogs that get a breed name (🟢) | 77.7% | 87.4% |
| …and that name is right | 96.9% | 95.4% |
| 🟡 dogs: true breed among the 3 shown | 94.2% | 93.5% |
| *For comparison, stage 2 (always naming a breed):* | | |
| Top-1 breed accuracy, all dogs | 89.2% | 90.3% |
| Top-3 breed accuracy, all dogs | 98.3% | 98.4% |

**How to read this:**
- In stage 2, the named breed was right about **90%** of the time and the app never admitted doubt.
- Now, for **87%** of dogs (with B2) it names a breed and is right about **95%** of the time. For the rest it honestly says "not sure" and shows 3 options, which include the right one about 94% of the time.
- **B2 is the better model here**, not because it's more accurate overall (90.3% vs 89.2%), but because it's **surer when it's right**. It names a breed for 87% of dogs instead of 78%, at the same accuracy target.
- **About 4–5% of cats still get through as dogs**, and nearly all of them are one breed. With B2, 51 test cats were accepted, and **45 of them were Sphynx cats**, the hairless breed. The model calls most of them "Mexican hairless", a hairless *dog* breed. For a model that learned "dog vs cat" partly from fur, a hairless cat really does look closer to a hairless dog. Raising the threshold wouldn't fix this without also rejecting real dogs.
- **About 2.5% of real dogs are rejected, and nearly all of them are Shiba Inus**: 55 of the roughly 60 rejected dogs. The Shiba Inu isn't one of ImageNet's 118 breeds. Its closest ImageNet class is **"dingo"**, a wild dog that sits just *outside* our dog range (class 273, after the wolves and the coyote). So the model does see a dog-like animal, but its probability goes to a class we don't count as a dog. The mistakes are not random: they come straight from a choice we made in stage 2 (which classes count as "dog").

**The plots:**
- [docs/stage3_thresholds_b2.png](docs/stage3_thresholds_b2.png) (also in the app, under *Experiment*).
- **Left:** the dog share of every test photo. Dogs pile up near 1, cats and food near 0, and the overlap in the middle is where the mistakes happen.
- **Right:** as the breed threshold goes up, the named breeds get more accurate (blue line), but fewer dogs get a name (dotted line). The threshold sits where the blue line reaches 95%.

**These numbers give the first real measurement of stage 2's model**, which earlier we could only estimate. Breed accuracy of about 90% top-1 and 98% top-3 matches the estimate. Stage 4 measures it properly on all 120 Stanford Dogs breeds.

## Concepts

- **The two kinds of mistakes.** Rejecting a real dog vs accepting a cat. One threshold trades one for the other, so you choose which mistake is worse for your users. This is the same as precision vs recall in classic ML.
- **Out-of-distribution input:** photos of things the model was never meant to handle. Real users *will* upload them.
- **Confidence is not correctness.** A "95% sure" model can still be wrong. The right plot shows how often it is right *at each confidence level*. Checking this is called calibration.
- **Selective prediction:** answering only when sure enough, and handing the rest to a fallback (here: "not sure, maybe one of these 3"). It's a common pattern in production AI: an assistant that escalates to a human, a fraud model that sends unclear cases for review.
- **Thresholds as configuration.** The values live in `config/stage3_thresholds.json`, not in the code, and are re-measured by a script. Change the rules, rerun the script, and the app follows. This is also the start of reproducibility.

## Try this in the app

1. Click the **six example test photos** under the upload box:
   - a pug (🟢 clear breed)
   - a beagle (🟡: beagle, Walker hound or English foxhound, three very similar hounds)
   - a Bengal cat (🔴)
   - a pizza (🔴)
   - **two mistakes:** a Shiba Inu (a real dog, wrongly 🔴) and a Sphynx cat (wrongly taken for a dog). Open *Under the hood* on these to see why.
2. Upload **your own photos**: your dog, a cat, a person, a cuddly toy, a drawing of a dog.
3. Open **Experiment** and move the sliders. The model doesn't run again, so the answer changes instantly.
   - Dog threshold at 0 is stage 2 again: every photo gets breeds.
   - Breed threshold at 0.95 means only very clear cases get a name.
4. Switch between **B2 and B0**. Each model has its own measured thresholds, so the sliders jump.

## Limitations

- **Breeds ImageNet doesn't know can be rejected as "no dog"**, like the Shiba Inu → "dingo". A quick fix would be to also count the wild dogs (dingo, dhole) as "dog", but then real wolves and dingoes would get a breed name. → better fixed in **stage 5** (fine-tuning, so the model learns these breeds).
- **Edge cases ImageNet half-knows:** wolves, foxes and hyenas are separate ImageNet classes, so they don't count as dogs. That's right for a fox, but a husky-like wolf may still get through as a husky.
- **Sphynx cats** pass as dogs. Few photos will be of hairless cats, so it's an acceptable mistake for this app. It shows that measuring tells you *which* mistakes you make, not just how many.
- **Cats are the only "hard" non-dogs we measured.** Cuddly toys, drawings and people weren't measured. Your own photos are the best test.
- **One photo, several dogs:** the dog share is high, but the breeds of different dogs get mixed. → a possible later stage (detection: find each dog first)
- **Other breeds ImageNet doesn't know** (Labradoodle, Kooikerhondje): the app may name a similar breed with confidence. → **Stage 5** (fine-tuning)

## Code

| File | What changed |
|---|---|
| `dogbreeds/decision.py` | New. `decide()` turns an analysis plus thresholds into 🟢/🟡/🔴. `breed_confidence()` computes the conditional probability. |
| `scripts/stage3_thresholds.py` | New. Measures the thresholds, writes the config and plots. Saves the model's scores in `data/` so plots can be redrawn without rerunning the models. |
| `config/stage3_thresholds.json` | New. The measured thresholds and test results per model. |
| `docs/stage3_thresholds_*.png` | New. The plots, one per model. |
| `dogbreeds/stages/stage3.py` | New. The stage 3 tab. The model runs once per photo and its result is kept in `gr.State`, so moving a slider only reapplies the thresholds. |
| `tests/test_decision.py` | New. The three outcomes, tested with made-up numbers, plus a check that the config is valid. |

**Rerun the measurement** (about 15 minutes the first time, seconds afterwards thanks to the saved scores):

```bash
uv run python scripts/stage3_thresholds.py
```

**Data** (in `data/`, not in git): Oxford-IIIT Pets is downloaded by torchvision, and pizza_steak_sushi is copied from the course project.
