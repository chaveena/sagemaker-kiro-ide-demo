"""Dataset subset preparation for the satellite land cover demo.

This module builds a small, per-class-capped subset of the EuroSAT dataset laid
out in an ``ImageFolder``-style directory tree so the SageMaker training job can
read it with a standard loader:

    data/
    |-- forest/    *.png
    |-- water/     *.png
    |-- cropland/  *.png
    |-- urban/     *.png

EuroSAT ships ten land-use categories. The demo maps four of those source
categories onto the four target classes and relabels each image to its target
class, yielding a four-class problem regardless of which source categories were
chosen:

    forest   <- Forest
    water    <- SeaLake
    cropland <- AnnualCrop
    urban    <- Industrial

The subset-preparation logic is kept separate from ``train.py`` so both the
notebook and the property tests can import it. The core function,
:func:`prepare_subset`, takes an injectable ``image_source`` abstraction so tests
can supply a synthetic in-memory source and never touch the network; the
notebook uses :func:`load_eurosat_source`, which is backed by torchvision.
"""

import os

from PIL import Image

# The four target classes the model predicts.
TARGET_CLASSES = ["forest", "water", "cropland", "urban"]

# Mapping from each target class to the EuroSAT source category that feeds it.
SOURCE_FOR = {
    "forest": "Forest",
    "water": "SeaLake",
    "cropland": "AnnualCrop",
    "urban": "Industrial",
}

# Default per-class cap. Small enough to keep the footprint and training quick,
# well under the size of any full EuroSAT category.
IMAGES_PER_CLASS = 200


def prepare_subset(image_source, out_dir, cap=IMAGES_PER_CLASS, holdout=2):
    """Build an ImageFolder-style subset from an injectable image source.

    For each target class in :data:`TARGET_CLASSES`, this function asks
    ``image_source`` for the images belonging to that class's EuroSAT source
    category, keeps at most ``cap`` of them, writes those images to
    ``out_dir/<target_class>/`` as PNG files named by index, and sets aside the
    first ``holdout`` of them (as in-memory references) for the post-training
    prediction grid.

    Per-class counts are computed by counting the image files actually written
    to each class directory, so the reported numbers reflect on-disk reality.

    Args:
        image_source: A callable implementing the source contract described
            below. It is invoked once per target class with that class's EuroSAT
            source category name.

            Contract::

                image_source(source_category_name: str) -> list[PIL.Image.Image]

            Given an EuroSAT source category name (e.g. ``"Forest"``,
            ``"SeaLake"``, ``"AnnualCrop"``, ``"Industrial"``), the source
            returns a list of ``PIL.Image.Image`` objects for that category. The
            order is up to the source; ``prepare_subset`` slices from the front.
            The source must not require network access to be *called* here --
            any download is the source's own concern (see
            :func:`load_eurosat_source`, which defers the download until first
            use). Returning fewer than ``cap`` images is allowed; the subset then
            contains only what was available.
        out_dir: Root directory for the subset. One subdirectory per target
            class is created beneath it. Created if it does not exist.
        cap: Maximum number of images to keep per class. Defaults to
            :data:`IMAGES_PER_CLASS`.
        holdout: Number of images per class to also set aside, in memory, into
            the returned held-out list. These references point at images that
            are part of the written subset; the held-out list itself is never
            written for its own sake and is not uploaded for training. Defaults
            to 2.

    Returns:
        A tuple ``(counts, held_out)`` where:
          - ``counts`` is a ``dict`` mapping each target class name to the number
            of image files written to its directory (on-disk truth).
          - ``held_out`` is a ``list`` of ``(PIL.Image.Image, target_label)``
            pairs -- the first ``holdout`` images of each class, kept in memory
            for the post-training predicted-vs-actual grid.
    """
    counts = {}
    held_out = []

    for target_class in TARGET_CLASSES:
        source_category = SOURCE_FOR[target_class]

        # Ask the injectable source for this category's images, then cap them.
        images = list(image_source(source_category))[:cap]

        # Set aside the first `holdout` as in-memory references for the grid.
        for img in images[:holdout]:
            held_out.append((img, target_class))

        # Write the capped images to out_dir/<target_class>/ as PNG by index.
        class_dir = os.path.join(out_dir, target_class)
        os.makedirs(class_dir, exist_ok=True)
        for index, img in enumerate(images):
            img.save(os.path.join(class_dir, f"{index}.png"))

        # Count the files actually on disk so reported counts match reality.
        counts[target_class] = _count_image_files(class_dir)

    return counts, held_out


def _count_image_files(class_dir):
    """Count image files present in a class directory.

    Only regular files with a recognized image extension are counted, so the
    reported per-class total reflects images actually written to disk.

    Args:
        class_dir: Path to a single class's directory.

    Returns:
        The number of image files in ``class_dir``.
    """
    image_extensions = (".png", ".jpg", ".jpeg")
    if not os.path.isdir(class_dir):
        return 0
    return sum(
        1
        for name in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, name))
        and name.lower().endswith(image_extensions)
    )


def load_eurosat_source(root="./eurosat_data"):
    """Return an ``image_source`` callable backed by torchvision's EuroSAT.

    This is the source the notebook uses. It downloads EuroSAT on first use
    (torchvision caches under ``root``) and groups the dataset's images by their
    EuroSAT class name, so the returned callable satisfies the ``image_source``
    contract expected by :func:`prepare_subset`::

        source(source_category_name: str) -> list[PIL.Image.Image]

    The download and grouping are performed lazily -- nothing touches the
    network until the returned callable is first invoked -- and the grouped
    result is cached so repeated calls are cheap.

    Args:
        root: Directory where torchvision stores / caches the EuroSAT download.

    Returns:
        A callable mapping an EuroSAT category name to the list of
        ``PIL.Image.Image`` objects for that category.
    """
    # Imported lazily so importing this module (and running the property tests)
    # never requires torchvision or a network connection.
    from torchvision import datasets

    cache = {"by_category": None}

    def _build_index():
        dataset = datasets.EuroSAT(root=root, download=True)
        # dataset.classes[i] is the EuroSAT category name for target index i.
        class_names = dataset.classes
        by_category = {name: [] for name in class_names}
        for image, target_index in dataset:
            by_category[class_names[target_index]].append(image)
        return by_category

    def source(source_category_name):
        if cache["by_category"] is None:
            cache["by_category"] = _build_index()
        return cache["by_category"].get(source_category_name, [])

    return source
