# Satellite Land Cover Demo

An end-to-end Amazon SageMaker demo that trains a small PyTorch CNN to classify
satellite tiles into four land-cover / land-use classes and then visualizes its
predictions. It is built to be a clear, working, top-to-bottom example of a
SageMaker managed training job launched with the **SageMaker Python SDK v3
`ModelTrainer`** — not a state-of-the-art model.

The four classes (a subset of the public [EuroSAT](https://github.com/phelber/EuroSAT)
Sentinel-2 dataset) are:

| Target class | EuroSAT source category |
| ------------ | ----------------------- |
| `forest`     | Forest                  |
| `water`      | SeaLake                 |
| `cropland`   | AnnualCrop              |
| `urban`      | Industrial              |

## What the notebook does

Open [`satellite_land_cover_demo.ipynb`](satellite_land_cover_demo.ipynb) and run
it top to bottom. The story runs in eight steps:

1. **Set up the SageMaker session** — resolves the execution role and default
   S3 bucket.
2. **Prepare a small EuroSAT subset** — downloads EuroSAT once (cached by
   torchvision), maps four source categories onto the four target classes, caps
   images per class, and sets aside a few held-out tiles in memory.
3. **Preview sample tiles** — a matplotlib grid of example images.
4. **Upload the subset to S3** — stages the training data under a demo prefix in
   the default bucket; this becomes the job's `training` input channel.
5. **Configure the `ModelTrainer`** — resolves the managed PyTorch training
   image, points `source_dir` at the project root (see *Source packaging*
   below), and sets compute and hyperparameters.
6. **Launch the managed training job** — submits the job **non-blocking**
   (`wait=False`) and polls `describe_training_job` to print each status
   transition (`Starting → Downloading → Training → Uploading → Completed`).
7. **Load the trained model back** — downloads `model.tar.gz`, rebuilds the
   `SmallCNN`, and loads the weights for local inference.
8. **Predicted vs. actual** — runs the model on the held-out tiles and draws an
   annotated grid.

## Project layout

| Path                                | Purpose |
| ----------------------------------- | ------- |
| `satellite_land_cover_demo.ipynb`   | The guided, top-to-bottom notebook. |
| `train.py`                          | SageMaker training entry script: `SmallCNN` architecture, arg parsing, training loop, and artifact saving. Also imported locally by the notebook/tests for the model definition. |
| `data_prep.py`                      | Builds the ImageFolder-style EuroSAT subset and the held-out tiles. |
| `viz.py`                            | Reusable inference + plotting helpers (`load_trained_model`, `predict_labels`, `show_prediction_grid`). |
| `reporting.py`                      | Pure, dependency-free helpers for reporting a job's result / failure reason. |
| `tests/`                            | Property-based unit tests and optional AWS smoke tests. |
| `requirements.txt`                  | Python dependencies. |

## Setup

Requires Python 3.10+ and AWS credentials with SageMaker + S3 access (in
SageMaker Studio / JupyterLab the execution role is resolved automatically).

```bash
pip install -r requirements.txt
```

Dependencies: `sagemaker>=3`, `torch`, `torchvision`, `matplotlib`.

## Key implementation notes

### Class-label ordering
Inference must map the model's output index back to a label using the **same
ordering the model was trained with**. `train.py` uses torchvision's
`ImageFolder`, which assigns class indices by sorting the class directory names
**alphabetically** (`cropland, forest, urban, water`) — *not* the
`TARGET_CLASSES` declaration order. `viz.py` therefore uses
`MODEL_INDEX_TO_LABEL = sorted(TARGET_CLASSES)`. Using the wrong order silently
mislabels every prediction (a permutation of the true labels).

### Source packaging
The training job reads its data from the `training` S3 input channel, so the
packaged code bundle must **not** contain the local dataset. The notebook uses:

```python
SourceCode(
    source_dir=".",
    entry_script="train.py",
    ignore_patterns=["data", "eurosat_data", "model", "model.tar.gz",
                     "tests", ".hypothesis", ".pytest_cache"],
)
```

`ignore_patterns` keeps the uploaded bundle to a few KB (just the `.py` sources)
so the job submits almost instantly. Without it, packaging `source_dir="."`
would tar and upload the whole directory — including the ~224 MB EuroSAT
download — on every run.

## Running the tests

Fast, offline unit tests (no AWS, no network):

```bash
pip install pytest hypothesis   # test-only dependencies (not in requirements.txt)
pytest -m "not slow"
```

The suite includes property-based tests (via `hypothesis`) for the model output
shape, argument parsing, data preparation, the prediction grid, and result
reporting.

Optional smoke tests that launch **real, long-running AWS work** (a managed
training job, S3 I/O) are marked `slow` and are skipped by default. Run them
explicitly only when you intend to incur AWS usage:

```bash
pytest -m slow
```

## Generated artifacts

Running the notebook creates local directories that are intentionally
git-ignored and safe to delete (they are regenerated on the next run):

- `data/` — the prepared ImageFolder subset
- `eurosat_data/` — torchvision's EuroSAT download cache
- `model/`, `model.tar.gz` — the downloaded/extracted trained model
