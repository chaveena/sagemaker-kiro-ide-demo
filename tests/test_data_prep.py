"""Property-based tests for the subset-prep logic in data_prep.py.

These tests use a synthetic, in-memory image source so they never touch the
network or download EuroSAT. The source is a callable satisfying the
``image_source`` contract that prepare_subset expects:

    image_source(source_category_name: str) -> list[PIL.Image.Image]
"""

import os
import sys
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

# Make the project root importable so `data_prep` resolves regardless of the
# directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_prep import (  # noqa: E402
    SOURCE_FOR,
    TARGET_CLASSES,
    prepare_subset,
)

TARGET_CLASS_SET = set(TARGET_CLASSES)
# The EuroSAT source category names the four target classes map to.
SOURCE_CATEGORIES = [SOURCE_FOR[c] for c in TARGET_CLASSES]


def make_synthetic_source(availabilities):
    """Build an in-memory ``image_source`` callable from a per-category count map.

    Args:
        availabilities: dict mapping EuroSAT source category name -> number of
            images that category should yield.

    Returns:
        A callable ``source(source_category_name) -> list[PIL.Image.Image]`` that
        generates that many small, color-varied RGB images per category.
    """

    def source(source_category_name):
        num = availabilities.get(source_category_name, 0)
        images = []
        for i in range(num):
            # Vary the color a little so the images are not all identical.
            color = ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256)
            images.append(Image.new("RGB", (64, 64), color))
        return images

    return source


def _count_files(path):
    """Count regular files in a directory (0 if it does not exist)."""
    if not os.path.isdir(path):
        return 0
    return sum(
        1 for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))
    )


# Strategy: a per-category availability map keyed by the four source categories.
def availabilities_strategy(min_count, max_count):
    return st.fixed_dictionaries(
        {cat: st.integers(min_value=min_count, max_value=max_count) for cat in SOURCE_CATEGORIES}
    )


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_subset_stays_small_and_capped(data):
    """Property 1: subset stays per-class capped and strictly under full size.

    **Feature: satellite-model-demo, Property 1: For any per-class cap N smaller
    than the smallest full EuroSAT category size, the prepared subset SHALL
    contain at most N images per class and a total strictly less than the full
    EuroSAT dataset size.**
    Validates: Requirements 2.2
    """
    # Choose a cap N, then make every category strictly larger than N so N is
    # smaller than the smallest full category.
    cap = data.draw(st.integers(min_value=1, max_value=50))
    availabilities = {
        cat: data.draw(st.integers(min_value=cap + 1, max_value=cap + 100))
        for cat in SOURCE_CATEGORIES
    }
    full_dataset_size = sum(availabilities.values())

    source = make_synthetic_source(availabilities)

    # A fresh directory per generated input, cleaned up automatically.
    with tempfile.TemporaryDirectory() as out_dir:
        counts, _held_out = prepare_subset(source, out_dir, cap=cap, holdout=2)

        # Each class holds at most N images.
        for target_class in TARGET_CLASSES:
            assert counts[target_class] <= cap

        # Total written is strictly less than the full dataset size (since every
        # category had strictly more than N images and we kept at most N each).
        total_written = sum(counts.values())
        assert total_written < full_dataset_size


@settings(max_examples=100, deadline=None)
@given(availabilities=availabilities_strategy(1, 20))
def test_label_domain_is_four_target_classes(availabilities):
    """Property 2: distinct class labels equal exactly the four target classes.

    **Feature: satellite-model-demo, Property 2: For any prepared subset, the set
    of distinct class labels SHALL equal exactly {forest, water, cropland,
    urban}.**
    Validates: Requirements 2.3
    """
    source = make_synthetic_source(availabilities)

    # A fresh directory per generated input, cleaned up automatically.
    with tempfile.TemporaryDirectory() as out_dir:
        counts, held_out = prepare_subset(source, out_dir, cap=10, holdout=2)

        # Reported count keys are exactly the four target classes.
        assert set(counts.keys()) == TARGET_CLASS_SET

        # On-disk class subdirectories are exactly the four target classes.
        on_disk_dirs = {
            name
            for name in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, name))
        }
        assert on_disk_dirs == TARGET_CLASS_SET

        # Held-out labels are drawn only from the four target classes.
        held_out_labels = {label for _img, label in held_out}
        assert held_out_labels <= TARGET_CLASS_SET


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_reported_counts_match_on_disk(data):
    """Property 3: reported per-class counts equal on-disk files; total is sum.

    **Feature: satellite-model-demo, Property 3: For any subset selection, the
    per-class counts reported SHALL equal the actual files written per class
    directory, and the reported total SHALL equal their sum.**
    Validates: Requirements 2.5
    """
    cap = data.draw(st.integers(min_value=1, max_value=30))
    # Availabilities span both below and above the cap.
    availabilities = {
        cat: data.draw(st.integers(min_value=0, max_value=cap + 20))
        for cat in SOURCE_CATEGORIES
    }

    source = make_synthetic_source(availabilities)

    # A fresh directory per generated input, cleaned up automatically.
    with tempfile.TemporaryDirectory() as out_dir:
        counts, _held_out = prepare_subset(source, out_dir, cap=cap, holdout=2)

        # Each reported count equals the actual number of files on disk.
        total_on_disk = 0
        for target_class in TARGET_CLASSES:
            class_dir = os.path.join(out_dir, target_class)
            files_on_disk = _count_files(class_dir)
            assert counts[target_class] == files_on_disk
            total_on_disk += files_on_disk

        # Reported total equals the sum across all class dirs.
        assert sum(counts.values()) == total_on_disk
