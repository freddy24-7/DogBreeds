# DogBreeds

Upload a photo of a dog and get its breed, built in small stages. After each stage the app runs and does something new.

## The idea

The model is one component behind a small interface, like a service you would mock in normal software:

```python
predictor.predict(image, top_k=3)  ->  [("beagle", 0.82), ("basset", 0.09), ("harrier", 0.03)]
```

The app (`app.py`) only calls this method. We start with a fake predictor and swap in real models later without touching the UI. The tests in `tests/` check this contract for every predictor, so they act as regression tests for each new model.

## Run it

```bash
uv run python app.py      # then open http://localhost:7870
uv run pytest             # run the tests
```

## Stages

| # | What you see in the app | What you learn | Status |
|---|---|---|---|
| 1 | Upload page with a top-3 breed list, random guesses | App structure, the `Predictor` interface | ✅ done |
| 2 | Real predictions without any training (ImageNet already knows ~120 breeds) | Pre-trained models, softmax, top-k | |
| 3 | "I don't see a dog" for cats, cars, ... | Confidence thresholds, out-of-distribution input | |
| 4 | Model report: accuracy on a Stanford Dogs test set | Datasets, DataLoaders, evaluation before training | |
| 5 | Model picker: ImageNet vs fine-tuned | Transfer learning, caching features | |
| 6 | "Commonly confused" breeds with examples | Error analysis, fine-tuning more layers | |
| 7 | Heatmap of where the model looked | Explainability (Grad-CAM) | |
| 8 | "Dogs that look like yours" | Embeddings, similarity search | |
| 9 | The app online | Deployment (Hugging Face Space) | |

## Layout

```
app.py                    the Gradio UI
dogbreeds/predictors.py   the Predictor interface and all predictors
tests/                    the contract every predictor must pass
```
