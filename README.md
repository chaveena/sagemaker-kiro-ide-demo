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
pytest -m "slow"
```

## From demo to production

This project is deliberately minimal so the SageMaker mechanics stay visible. A
production-grade training pipeline keeps the same `ModelTrainer` shape but
changes almost every parameter. The table below maps each demo choice to what
you would reconsider, followed by notes on the bigger topics.

| Concern | This demo | Production direction |
| ------- | --------- | -------------------- |
| Data volume | 800 tiles, 4 classes | Full dataset(s), often 100s of GB–TBs |
| Data upload | per-file `s3.upload_file` loop | Bulk transfer + a manifest; `FastFile`/`Pipe` input mode |
| Instance | `ml.m5.large` (CPU) | Right-sized GPU (e.g. `ml.g5.*` / `ml.p4d.*`) |
| Instance count | `instance_count=1` | Multi-instance distributed training when justified |
| Hyperparameters | fixed `epochs=3, batch-size=32` | Tuned via SageMaker Automatic Model Tuning |
| Model | from-scratch `SmallCNN` | Pretrained backbone + fine-tuning |
| Runtime cap | `max_runtime=3600s` | Sized to the real job; checkpointing enabled |
| Cost | on-demand | Managed Spot training with checkpointing |
| Validation | none (train only) | Held-out val/test split + metrics emitted to the job |

### Managing a much larger dataset
- **Don't upload file-by-file.** The demo's `s3.upload_file` loop is fine for
  800 tiles but pathological for millions. Use `aws s3 sync`, S3 Batch
  Operations, or write the data as a few large shards (e.g. WebDataset tars,
  TFRecord/RecordIO, or Parquet) so you move a handful of big objects instead of
  millions of tiny ones.
- **Change the input mode.** The demo uses the default `File` mode, which
  downloads the *entire* channel to the instance's EBS volume before training
  starts — impractical past a certain size. For large data use **`FastFile`**
  (streams objects on demand, no full download) or **`Pipe`** mode, set via the
  channel's `input_mode`. For very large or many-epoch workloads, an
  **FSx for Lustre** file system linked to your S3 bucket gives high-throughput,
  low-latency access and avoids re-downloading each run.
- **Keep the `ImageFolder`-on-disk contract only if it still fits.** At scale,
  reading millions of individual files is slow; prefer a sharded format with a
  streaming `Dataset` in `train.py`.
- **Version your data.** Track dataset versions/prefixes (or use SageMaker
  Feature Store / a data catalog) so runs are reproducible.

### Choosing a much larger instance
- **Match the accelerator to the model.** CPU (`ml.m5.large`) suits this tiny
  net. Real CNNs/transformers want GPUs — e.g. `ml.g5.xlarge` for single-GPU
  fine-tuning up through `ml.p4d.24xlarge` / `ml.p5.*` for large multi-GPU jobs.
  Check the model + batch fit in GPU memory first.
- **Scale up before scaling out.** A single bigger instance (more/bigger GPUs)
  is simpler and often cheaper than multiple instances. Only set
  `instance_count > 1` when one instance can't hold the model or the throughput
  gain clearly beats the communication overhead; that also means adding a
  distributed strategy (DDP / FSDP, or SageMaker's distributed libraries) in
  `train.py`.
- **Mind the quotas and the bill.** GPU instance types have per-account service
  quotas you may need raised, and they are expensive — see cost notes below.

### Training robustness and cost
- **Checkpointing.** Long jobs should periodically save state to a checkpoint S3
  path (SageMaker syncs a local checkpoint dir to S3) so a job can resume rather
  than restart. This is also what makes Spot safe.
- **Managed Spot training** can cut cost substantially in exchange for possible
  interruptions; combined with checkpointing, interruptions just resume.
- **Right-size the runtime cap.** `max_runtime_in_seconds=3600` here is a demo
  guardrail; set it to a realistic ceiling for the real job.

### Hyperparameters, metrics, and model quality
- **Automatic Model Tuning.** Instead of hardcoding `epochs`/`batch-size`, define
  search ranges and an objective metric and let SageMaker run a tuning job.
- **Emit metrics.** Have `train.py` log validation metrics in a parseable form
  and register them as SageMaker **metric definitions** so they appear in the
  console and can drive tuning/early stopping. The demo trains only and reports
  no validation metric.
- **Use a proper split** (train/validation/test) — pass them as separate input
  channels (`validation`, `test`) alongside `training`.

### MLOps, security, and governance
- **Pipelines over notebooks.** Move the flow into a **SageMaker Pipeline**
  (processing → training → evaluation → conditional register) so it is
  repeatable and auditable, rather than run cell-by-cell.
- **Model Registry + deployment.** Register the resulting model version and
  deploy behind a SageMaker Endpoint (real-time) or use Batch Transform, rather
  than only downloading `model.tar.gz` for local inspection as the demo does.
- **Least-privilege IAM.** The demo relies on the notebook execution role;
  production should scope roles to the specific S3 prefixes and actions needed.
- **Encryption & isolation.** Enable S3/EBS encryption (KMS), run jobs in a VPC
  with no direct internet egress, and enable network isolation for the training
  container where appropriate.
- **Reproducibility.** Pin the framework/container image and dependency
  versions, and record the data version, hyperparameters, and git commit for
  each run.

> Cost note: GPU/multi-instance training and endpoints can be expensive. For
> current pricing and estimates, use the
> [AWS Pricing Calculator](https://calculator.aws/) and the SageMaker pricing
> page rather than assuming figures.

## Generated artifacts

Running the notebook creates local directories that are intentionally
git-ignored and safe to delete (they are regenerated on the next run):

- `data/` — the prepared ImageFolder subset
- `eurosat_data/` — torchvision's EuroSAT download cache
- `model/`, `model.tar.gz` — the downloaded/extracted trained model
