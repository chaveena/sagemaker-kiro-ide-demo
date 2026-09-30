"""Integration/smoke test: managed training job via ModelTrainer (Task 8.2).

Validates:
  * Requirement 4.1 -- a ModelTrainer is configured with train.py as the entry
    script against a managed PyTorch training image.
  * Requirement 4.2 / 4.3 -- the job runs on managed infrastructure with a
    named "training" S3 input channel (not in this process).
  * Requirement 4.5 -- the model artifact S3 URI is read from the training job
    description (ModelArtifacts.S3ModelArtifacts), the SDK-v3 replacement for
    the removed v2 ``estimator.model_data``.

This launches a REAL, billable SageMaker managed training job, so it is
double-gated and OFF by default. It runs only when BOTH ``RUN_AWS_SMOKE_TESTS=1``
AND ``RUN_AWS_TRAINING_SMOKE=1`` are set and boto3 can resolve a region and
credentials; otherwise it skips cleanly (see ``tests/conftest.py``). It is also
marked ``@pytest.mark.slow`` because it waits for the training job to complete.

The test generates a tiny synthetic EuroSAT-like subset (a few 64x64 RGB PNGs
across the four class dirs forest/water/cropland/urban) -- it never downloads
real EuroSAT -- uploads it to S3, and runs a 1-epoch job to keep it short.
"""

import os
import sys
import uuid
import shutil
import tempfile

import pytest
from PIL import Image

# Make the project root importable so `conftest` helpers and project modules
# resolve regardless of the directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conftest import (  # noqa: E402
    aws_training_smoke_enabled,
    aws_training_smoke_reason,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not aws_training_smoke_enabled(),
        reason=aws_training_smoke_reason() or "AWS training smoke disabled",
    ),
]

# The four target classes; matches the notebook / data_prep layout.
TARGET_CLASSES = ["forest", "water", "cropland", "urban"]


def _make_tiny_eurosat_like_subset(root, per_class=3):
    """Write a tiny ImageFolder-style subset of 64x64 RGB PNGs, four classes.

    Purely synthetic (solid-color tiles) -- no real EuroSAT download. Enough
    images per class so a 1-epoch DataLoader run has something to train on.
    """
    for class_index, class_name in enumerate(TARGET_CLASSES):
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for index in range(per_class):
            color = (class_index * 60, index * 40, 128)
            Image.new("RGB", (64, 64), color=color).save(
                os.path.join(class_dir, f"{index}.png")
            )


def test_managed_training_job_completes_and_reports_artifact():
    """A managed training job completes and reports its model artifact URI.

    Builds a ModelTrainer exactly as the notebook does (managed pytorch 2.2 /
    py310 / ml.m5.large training image; SourceCode(source_dir=".",
    entry_script="train.py"); Compute(instance_count=1); small hyperparameters),
    uploads a tiny synthetic subset to S3, and calls ``train()`` with an
    ``InputData`` channel named "training". After completion it uses boto3
    ``describe_training_job`` and asserts the status is "Completed" and the
    model artifact is a non-empty ``s3://`` URI (Req 4.1, 4.2, 4.3, 4.5).
    """
    import boto3
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import (
        Compute,
        InputData,
        OutputDataConfig,
        SourceCode,
    )

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    session = Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    role = get_execution_role()
    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")

    instance_type = "ml.m5.large"
    key_prefix = f"satellite-land-cover-demo-smoke/{uuid.uuid4().hex}/data"
    local_dir = tempfile.mkdtemp(prefix="training_smoke_")
    uploaded_keys = []

    try:
        _make_tiny_eurosat_like_subset(local_dir)

        # Upload the synthetic subset to S3 (the training input channel).
        for root, _dirs, files in os.walk(local_dir):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{key_prefix}/" + rel_path.replace(os.sep, "/")
                s3.upload_file(local_path, bucket, s3_key)
                uploaded_keys.append(s3_key)
        s3_data_uri = f"s3://{bucket}/{key_prefix}"

        # Resolve the managed PyTorch training image (Req 4.1).
        training_image = image_uris.retrieve(
            framework="pytorch",
            region=region,
            version="2.2",
            py_version="py310",
            instance_type=instance_type,
            image_scope="training",
        )

        # Ship train.py from the project root as the entry script (Req 4.1).
        source_code = SourceCode(
            source_dir=project_root, entry_script="train.py"
        )
        compute = Compute(instance_type=instance_type, instance_count=1)

        model_trainer = ModelTrainer(
            training_image=training_image,
            role=role,
            source_code=source_code,
            compute=compute,
            # 1 epoch / small batch to keep the real job short.
            hyperparameters={"epochs": 1, "batch-size": 8},
            output_data_config=OutputDataConfig(
                s3_output_path=f"s3://{bucket}/"
            ),
        )

        # Launch on managed infra with the named "training" channel (4.2, 4.3).
        model_trainer.train(
            input_data_config=[
                InputData(channel_name="training", data_source=s3_data_uri),
            ]
        )

        # Artifact + status come from the job description in SDK v3 (Req 4.5).
        job_name = model_trainer._latest_training_job.training_job_name
        desc = sm.describe_training_job(TrainingJobName=job_name)

        assert desc["TrainingJobStatus"] == "Completed"
        model_artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
        assert model_artifact, "expected a non-empty model artifact URI"
        assert model_artifact.startswith("s3://")
    finally:
        # Best-effort cleanup of the uploaded input objects.
        for key in uploaded_keys:
            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except Exception:
                pass
        shutil.rmtree(local_dir, ignore_errors=True)
