"""SageMaker training entry point for the satellite land cover demo.

This script defines a small convolutional neural network that classifies
64x64 RGB satellite tiles into four land cover / land use classes
(forest, water, cropland, urban). The model is trained from scratch with no
pretrained weights.

-----------------------------------------------------------------------------
NEW TO SAGEMAKER? READ THIS FIRST.
-----------------------------------------------------------------------------
Almost everything below is ordinary PyTorch. What makes this a *SageMaker*
training script is a small contract between this file and the managed training
container that SageMaker spins up on your behalf. You do not run this script
yourself; SageMaker runs it inside a container on a separate machine, and it
communicates with your code through **environment variables** and a
**filesystem convention** -- not through any SageMaker Python API. That means
this file has zero ``import sagemaker`` / ``import boto3``; it only reads a few
env vars. The whole contract is:

  1. INPUT DATA arrives as a "channel." When the notebook launches the job with
     an input channel named ``training``, SageMaker downloads that data from S3
     onto the container's disk and tells you where via the environment variable
     ``SM_CHANNEL_TRAINING``. (A channel named ``foo`` -> ``SM_CHANNEL_FOO``.)

  2. HYPERPARAMETERS arrive as COMMAND-LINE ARGUMENTS. The ``hyperparameters={...}``
     dict passed to ``ModelTrainer`` in the notebook is handed to this script as
     CLI flags (e.g. ``--epochs 3``), which is why we parse them with argparse.

  3. THE MODEL OUTPUT goes in ``SM_MODEL_DIR``. Whatever you write into the
     directory named by the ``SM_MODEL_DIR`` env var is automatically packaged
     into ``model.tar.gz`` and uploaded to S3 when the job finishes. Writing
     the weights anywhere else means they are lost.

  4. LOGS are just stdout. Anything you ``print`` is captured into CloudWatch
     Logs, so print statements are how you watch a remote job's progress.

Everything else -- the model, the DataLoader, the training loop -- is exactly
what you would write for local PyTorch training.
-----------------------------------------------------------------------------
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args(argv=None):
    """Parse command-line arguments for the training entry point.

    SageMaker passes hyperparameters as CLI arguments and sets the
    ``SM_CHANNEL_TRAINING`` and ``SM_MODEL_DIR`` environment variables, which
    are used as defaults for ``--data-dir`` and ``--model-dir`` respectively so
    the managed training environment is honored automatically.

    Args:
        argv: Optional list of argument strings. When ``None`` (the default),
            arguments are read from ``sys.argv``. Passing an explicit list makes
            the parser easy to exercise from tests.

    Returns:
        argparse.Namespace with ``epochs``, ``batch_size``, ``data_dir``, and
        ``model_dir`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train a small CNN on satellite tiles."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        # SAGEMAKER: SM_CHANNEL_TRAINING is the on-disk path where SageMaker has
        # placed the data from the "training" input channel (downloaded from S3
        # before this script runs). Using it as the default means the script
        # "just works" in the container, while still allowing a manual path
        # (e.g. for a local test run) to override it.
        default=os.environ.get("SM_CHANNEL_TRAINING"),
        help="Input data path (defaults to the SM_CHANNEL_TRAINING env var).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        # SAGEMAKER: SM_MODEL_DIR is the directory SageMaker packages into
        # model.tar.gz and uploads to S3 after training. Saving the weights here
        # (see save()) is what makes the trained model persist beyond the job.
        default=os.environ.get("SM_MODEL_DIR"),
        help="Model output path (defaults to the SM_MODEL_DIR env var).",
    )
    return parser.parse_args(argv)


class SmallCNN(nn.Module):
    """A small from-scratch CNN for 3x64x64 RGB tiles.

    Architecture:
        Input: 3 x 64 x 64
        Conv(3 -> 16, 3x3, pad=1) -> ReLU -> MaxPool(2)   # 16 x 32 x 32
        Conv(16 -> 32, 3x3, pad=1) -> ReLU -> MaxPool(2)  # 32 x 16 x 16
        Conv(32 -> 64, 3x3, pad=1) -> ReLU -> MaxPool(2)  # 64 x 8 x 8
        Flatten
        Linear(64*8*8 -> 128) -> ReLU
        Linear(128 -> num_classes)

    The final layer emits exactly ``num_classes`` logits (one per class).
    """

    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_loader(data_dir, batch_size):
    """Build a DataLoader over an ImageFolder-style dataset.

    The dataset directory is expected to contain one subdirectory per class,
    each holding image files (the layout torchvision's ``ImageFolder`` reads).
    Tiles are converted to tensors; EuroSAT tiles are already 64x64 RGB, but a
    ``Resize((64, 64))`` is included so any slightly off-size image is coerced
    to the shape the model expects.

    Args:
        data_dir: Path to the ImageFolder-style dataset root.
        batch_size: Mini-batch size for the returned loader.

    Returns:
        A ``torch.utils.data.DataLoader`` yielding ``(images, labels)`` batches
        with shuffling enabled.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, loader, epochs):
    """Train ``model`` on ``loader`` for ``epochs`` epochs.

    Uses cross-entropy loss and the Adam optimizer. The average loss for each
    epoch is printed so progress is visible in the training job logs.

    Args:
        model: The ``SmallCNN`` (or compatible) model to train in place.
        loader: A DataLoader yielding ``(images, labels)`` batches.
        epochs: Number of passes to make over the dataset.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / batches if batches else 0.0
        # SAGEMAKER: stdout from the container is streamed to CloudWatch Logs,
        # so a plain print is how per-epoch progress becomes visible for a
        # remote job you are not running interactively.
        print(f"Epoch {epoch + 1}/{epochs} - average loss: {avg_loss:.4f}")


def save(model, model_dir):
    """Save the trained model's weights to ``model_dir/model.pth``.

    The directory is created if it does not already exist.

    SAGEMAKER: ``model_dir`` is ``SM_MODEL_DIR`` (see ``parse_args``). After the
    script exits, SageMaker tars the *entire contents* of this directory into
    ``model.tar.gz`` and uploads it to the job's S3 output location. Anything
    saved elsewhere on the container disk is discarded when the job ends, so the
    model must be written here to survive. This is standard PyTorch otherwise --
    ``torch.save`` of a ``state_dict`` -- only the destination is special.

    Args:
        model: The trained model whose ``state_dict`` is saved.
        model_dir: Destination directory for the ``model.pth`` artifact.
    """
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))


def main():
    """Training entry point: parse args, load data, train, and save."""
    args = parse_args()
    loader = build_loader(args.data_dir, args.batch_size)
    model = SmallCNN(num_classes=4)
    train(model, loader, epochs=args.epochs)
    save(model, args.model_dir)


if __name__ == "__main__":
    main()
