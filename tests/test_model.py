"""Property-based tests for the SmallCNN model in train.py.

Feature: satellite-model-demo, Property 4: For any input batch of B valid
3x64x64 image tensors, the model's forward pass SHALL produce an output of
shape (B, 4).
Validates: Requirements 3.3
"""

import os
import sys

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Make the project root importable so `train` resolves regardless of the
# directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import SmallCNN  # noqa: E402


@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=16))
def test_forward_output_shape(batch_size):
    """Property 4: forward pass on a (B, 3, 64, 64) batch yields shape (B, 4).

    Feature: satellite-model-demo, Property 4: For any input batch of B valid
    3x64x64 image tensors, the model's forward pass SHALL produce an output of
    shape (B, 4).
    Validates: Requirements 3.3
    """
    model = SmallCNN(num_classes=4)
    model.eval()

    x = torch.randn(batch_size, 3, 64, 64)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, 4)
