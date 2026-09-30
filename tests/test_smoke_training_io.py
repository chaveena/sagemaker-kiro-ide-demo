"""Integration/smoke test: train.py honors the SageMaker training env (Task 8.3).

Validates:
  * Requirement 3.4 -- ``--data-dir`` defaults from the ``SM_CHANNEL_TRAINING``
    environment variable that SageMaker sets for the named input channel.
  * Requirement 3.5 -- ``--model-dir`` defaults from the ``SM_MODEL_DIR``
    environment variable, and a full local training run writes ``model.pth``
    into that directory (the file SageMaker collects into ``model.tar.gz``).

This test needs NO AWS: it emulates the SageMaker training-env contract locally
by setting ``SM_CHANNEL_TRAINING`` / ``SM_MODEL_DIR`` to temp directories, so it
runs normally in every environment (no skip). ``train.py`` is imported fresh so
its argparse defaults read the env vars set for this test.
"""

import os
import sys
import shutil
import tempfile
import importlib

from PIL import Image

# Make the project root importable so `train` resolves regardless of the
# directory pytest is invoked from (matches tests/test_model.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_tiny_imagefolder(root, per_class=4):
    """Write a tiny 2-class ImageFolder-style subset of 64x64 RGB PNGs."""
    for class_index, class_name in enumerate(["forest", "water"]):
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for index in range(per_class):
            color = (class_index * 80, index * 30, 120)
            Image.new("RGB", (64, 64), color=color).save(
                os.path.join(class_dir, f"{index}.png")
            )


def test_train_honors_sm_env_and_writes_artifact():
    """train.py reads SM_CHANNEL_TRAINING / SM_MODEL_DIR and writes model.pth.

    Sets the SageMaker training-env variables to temp dirs, confirms
    ``parse_args([])`` picks them up as ``--data-dir`` / ``--model-dir`` defaults
    (Req 3.4 / 3.5), then runs the real local pipeline (build_loader -> train
    for 1 epoch -> save) and asserts ``model.pth`` lands in the model dir.
    """
    data_dir = tempfile.mkdtemp(prefix="sm_channel_")
    model_dir = tempfile.mkdtemp(prefix="sm_model_")
    saved_env = {
        "SM_CHANNEL_TRAINING": os.environ.get("SM_CHANNEL_TRAINING"),
        "SM_MODEL_DIR": os.environ.get("SM_MODEL_DIR"),
    }

    try:
        _make_tiny_imagefolder(data_dir)

        # Set the SageMaker-provided env vars, then (re)import train so its
        # argparse defaults are computed against these values.
        os.environ["SM_CHANNEL_TRAINING"] = data_dir
        os.environ["SM_MODEL_DIR"] = model_dir
        import train  # noqa: E402
        train = importlib.reload(train)

        # Req 3.4 / 3.5: defaults come from the SageMaker env vars.
        args = train.parse_args([])
        assert args.data_dir == data_dir
        assert args.model_dir == model_dir

        # A full local run reads from data_dir and writes model.pth to model_dir.
        loader = train.build_loader(args.data_dir, batch_size=2)
        model = train.SmallCNN(num_classes=4)
        train.train(model, loader, epochs=1)
        train.save(model, args.model_dir)

        assert os.path.isfile(os.path.join(model_dir, "model.pth"))
    finally:
        # Restore the original environment and clean up temp dirs.
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(model_dir, ignore_errors=True)
