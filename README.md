# DogBreeds

Upload a photo of a dog and get its breed, built in small stages. After each stage the app runs and does something new.

## The idea

The model is one component behind a small interface, like a service you would mock in normal software:

```python
predictor.predict(image, top_k=3)  ->  [("beagle", 0.82), ("basset", 0.09), ("harrier", 0.03)]
```

The app (`app.py`) only calls this method. We start with a fake predictor and swap in real models later without touching the UI.

Each stage has its own **tab** in the app, so you can compare how the app behaved at every stage. The newest stage opens first. Each stage also has its own README with the details: `README_stage2.md`, and so on. The tests in `tests/` check this contract for every predictor, so they act as regression tests for each new model.

## Run it

```bash
uv run python app.py      # then open http://localhost:7870
uv run pytest             # run the tests
```

## Stages

| # | What you see in the app | What you learn | Status |
|---|---|---|---|
| 1 | Upload page with a top-3 breed list, random guesses | App structure, the `Predictor` interface | ✅ done |
| 2 | Real predictions without any training (ImageNet already knows 118 breeds) | Pre-trained models, softmax, top-k | ✅ done, see [README_stage2.md](README_stage2.md) |
| 3 | "I don't see a dog" for cats, cars, ... and "not sure which breed" | Measured thresholds, the two kinds of mistakes, out-of-distribution input | ✅ done, see [README_stage3.md](README_stage3.md) |
| 4 | Model report: accuracy on a Stanford Dogs test set | Datasets, DataLoaders, evaluation, calibration, leakage | ✅ done, see [README_stage4.md](README_stage4.md) |
| 5 | Our own head next to ImageNet's, and an honest comparison | Transfer learning, caching features, the training loop, validation | ✅ done (not shipped), see [README_stage5.md](README_stage5.md) |
| 6 | "Commonly confused" breeds with examples | Error analysis, fine-tuning more layers | |
| 7 | Heatmap of where the model looked | Explainability (Grad-CAM) | |
| 8 | "Dogs that look like yours" | Embeddings, similarity search | |
| 9 | The app online | Deployment (Hugging Face Space) | |

## Layout

```
app.py                    puts the stage tabs together into one app
dogbreeds/predictors.py   the Predictor interface and all predictors (models)
dogbreeds/stages/         one module per stage; each builds its own tab
tests/                    the contract every predictor must pass, plus app checks
README_stageN.md          what each stage adds and the concepts behind it
dogbreeds/data.py         datasets: Stanford Dogs, Oxford Pets dogs, your own photos
dogbreeds/evaluate.py     the metrics (top-k, per breed, confusions, calibration)
dogbreeds/finetune.py     training our own head on a frozen backbone (stage 5)
scripts/                  measurements that produce config/, reports/ and docs/
reports/                  model reports (stage 4: the baseline to beat)
models/                   trained weights (stage 5's head; not in git, made by scripts/)
config/, docs/            measured settings the app reads, and their plots
data/                     downloaded datasets (not in git)
```
