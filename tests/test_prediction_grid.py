"""Property-based test for the prediction grid inference core in viz.py.

Feature: satellite-model-demo, Property 6: For any held-out set of K tiles and
any model output of shape (K, 4), the predicted-vs-actual grid SHALL annotate
exactly K tiles, and each annotated predicted label SHALL be one of the four
target classes {forest, water, cropland, urban}.
Validates: Requirements 7.1, 7.3
"""

import os
import sys

from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from torchvision import transforms

# Make the project root importable so `viz` resolves regardless of the
# directory pytest is invoked from. Importing viz must not require sagemaker or
# boto3.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import viz  # noqa: E402
from train import SmallCNN  # noqa: E402

# A torchvision transform mapping a PIL image to a (3, 64, 64) tensor, matching
# the (1, 3, 64, 64) input contract predict_labels feeds the model.
_TO_TENSOR = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)


@settings(max_examples=100, deadline=None)
@given(
    k=st.integers(min_value=1, max_value=12),
    actual_labels=st.data(),
)
def test_prediction_grid_annotates_every_tile_with_valid_class(k, actual_labels):
    """Property 6: predict_labels yields K results, each with a valid class.

    Feature: satellite-model-demo, Property 6: For any held-out set of K tiles
    and any model output of shape (K, 4), the predicted-vs-actual grid SHALL
    annotate exactly K tiles, and each annotated predicted label SHALL be one of
    the four target classes {forest, water, cropland, urban}.
    Validates: Requirements 7.1, 7.3
    """
    # Real SmallCNN so the (K, 4) output contract is genuinely exercised:
    # predict_labels runs one image at a time, producing (1, 4) logits.
    model = SmallCNN(num_classes=4)
    model.eval()

    # Build K (image, actual_label) held-out pairs. Actual labels are arbitrary;
    # predict_labels does not depend on their correctness.
    held_out = []
    for _ in range(k):
        img = Image.new("RGB", (64, 64))
        actual_label = actual_labels.draw(st.sampled_from(viz.TARGET_CLASSES))
        held_out.append((img, actual_label))

    results = viz.predict_labels(model, held_out, _TO_TENSOR)

    # Exactly K tiles annotated, one per held-out tile.
    assert len(results) == k

    # Every predicted label is one of the four target classes.
    for _img, predicted_label, _actual_label in results:
        assert predicted_label in viz.TARGET_CLASSES
