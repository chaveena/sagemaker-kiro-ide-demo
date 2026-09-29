"""Property-based tests for the argument parser in train.py.

Feature: satellite-model-demo, Property 5: For any non-negative integer values
supplied for --epochs and --batch-size, the parser SHALL return those exact
values.
Validates: Requirements 3.6
"""

import os
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

# Make the project root importable so `train` resolves regardless of the
# directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import parse_args  # noqa: E402


@settings(max_examples=100, deadline=None)
@given(
    epochs=st.integers(min_value=0, max_value=100000),
    batch_size=st.integers(min_value=0, max_value=100000),
)
def test_hyperparameter_round_trip(epochs, batch_size):
    """Property 5: parsed --epochs and --batch-size equal the supplied values.

    Feature: satellite-model-demo, Property 5: For any non-negative integer
    values supplied for --epochs and --batch-size, the parser SHALL return
    those exact values.
    Validates: Requirements 3.6
    """
    args = parse_args(
        [
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--data-dir",
            "/tmp/d",
            "--model-dir",
            "/tmp/m",
        ]
    )

    assert args.epochs == epochs
    assert args.batch_size == batch_size
