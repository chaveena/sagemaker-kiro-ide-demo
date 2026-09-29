"""SageMaker training entry point for the satellite land cover demo.

This script defines a small convolutional neural network that classifies
64x64 RGB satellite tiles into four land cover / land use classes
(forest, water, cropland, urban). The model is trained from scratch with no
pretrained weights.

Additional pieces of the training entry point (command-line argument parsing,
data loading, the training loop, and artifact saving) are added in later tasks
and are intentionally not implemented here.
"""

import argparse
import os

import torch.nn as nn


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
        default=os.environ.get("SM_CHANNEL_TRAINING"),
        help="Input data path (defaults to the SM_CHANNEL_TRAINING env var).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
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
