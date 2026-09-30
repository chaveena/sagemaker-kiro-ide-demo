"""Reusable visualization and inference helpers for the satellite land cover demo.

These helpers are factored out of the guided notebook so the testable core
(model reconstruction, inference, and grid annotation) lives in importable,
unit- and property-testable functions rather than inside notebook cells.

Design constraints:
    - Importing this module must NOT require ``sagemaker``, ``boto3``, or any
      network access. Only ``torch`` / ``torchvision`` (and ``matplotlib``,
      imported lazily inside the drawing helpers) are used here. Any S3 access
      is injected via a ``download_fn`` callable supplied by the notebook.
    - ``predict_labels`` is the pure, side-effect-free core used by tests: it
      runs the model over the held-out tiles and returns structured results
      without drawing anything.

The four target classes are re-exposed from :mod:`data_prep` so there is a
single source of truth for the label domain.
"""

import os
import tarfile

import torch

# The four target classes the label domain is drawn from.
from data_prep import TARGET_CLASSES

# The order in which the trained model emits its logits. The training job uses
# torchvision's ImageFolder, which assigns class indices by SORTING the class
# directory names alphabetically -- NOT the TARGET_CLASSES order. Inference must
# map argmax back to a label using this same sorted order, otherwise every
# prediction is silently mislabeled (a permutation of the true labels).
MODEL_INDEX_TO_LABEL = sorted(TARGET_CLASSES)

# Import the model architecture from the training entry point so the model
# rebuilt for in-notebook inference is byte-for-byte the same network the
# managed training job trained.
from train import SmallCNN


def show_sample_grid(samples):
    """Draw a matplotlib grid of sample tiles, each titled with its label.

    Intended for the "before training" preview: it shows the kind of imagery
    the model learns from. Runs entirely on already-local images and never
    touches the training job.

    Args:
        samples: A list of ``(image, label)`` pairs, where ``image`` is a
            ``PIL.Image.Image`` and ``label`` is the class-label string to title
            the tile with.
    """
    import matplotlib  # noqa: F401  (ensure backend is initialized)
    import matplotlib.pyplot as plt

    n = len(samples)
    cols = 4
    rows = (n + cols - 1) // cols if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # axes may be a single Axes (1x1) or an ndarray; normalize to a flat list.
    flat_axes = _flatten_axes(axes)

    for ax, (img, label) in zip(flat_axes, samples):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    # Turn off any unused axes so empty tiles are not drawn.
    for ax in flat_axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def load_trained_model(model_data_uri, download_fn, extract_dir="model"):
    """Download, extract, and reconstruct the trained model for local inference.

    This brings the managed training job's artifact back into the notebook
    purely for visualization; it does not alter the training job.

    Args:
        model_data_uri: The S3 URI of the ``model.tar.gz`` artifact (typically
            ``estimator.model_data``).
        download_fn: A callable ``download_fn(model_data_uri, local_tar_path)``
            that fetches the artifact from ``model_data_uri`` and writes it to
            ``local_tar_path``. Injected so this function is testable without S3
            or network access.
        extract_dir: Directory into which ``model.tar.gz`` is extracted. The
            saved weights are expected at ``<extract_dir>/model.pth``. Defaults
            to ``"model"``.

    Returns:
        A ``SmallCNN(num_classes=4)`` instance with the saved ``state_dict``
        loaded (mapped to CPU) and set to eval mode.
    """
    local_tar_path = "model.tar.gz"
    download_fn(model_data_uri, local_tar_path)

    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(local_tar_path) as tar:
        tar.extractall(extract_dir)

    model = SmallCNN(num_classes=4)
    state_dict = torch.load(
        os.path.join(extract_dir, "model.pth"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_labels(model, held_out, to_tensor):
    """Run the model on each held-out tile and return predicted/actual labels.

    This is the pure, testable core of the post-training visualization: it
    performs inference but draws nothing. Inference runs under
    ``torch.no_grad()`` and the predicted label is the target class at the
    argmax of the model's logits.

    Args:
        model: A model whose forward pass maps a ``(1, 3, 64, 64)`` tensor to a
            ``(1, 4)`` logit tensor (e.g. the result of
            :func:`load_trained_model`).
        held_out: A list of ``(image, actual_label)`` pairs, where ``image`` is
            a ``PIL.Image.Image`` and ``actual_label`` is the true class-label
            string.
        to_tensor: A callable mapping a ``PIL.Image.Image`` to a
            ``(3, 64, 64)`` tensor (e.g. a torchvision transform).

    Returns:
        A list of ``(image, predicted_label, actual_label)`` tuples, one per
        held-out tile and in the same order. ``predicted_label`` is always one
        of :data:`TARGET_CLASSES`.
    """
    results = []
    with torch.no_grad():
        for img, actual_label in held_out:
            tensor = to_tensor(img).unsqueeze(0)
            logits = model(tensor)
            predicted_index = int(logits.argmax(dim=1))
            # Map the logit index back to a label using the SAME ordering the
            # model was trained with (ImageFolder's alphabetical class order),
            # not the TARGET_CLASSES declaration order.
            predicted_label = MODEL_INDEX_TO_LABEL[predicted_index]
            results.append((img, predicted_label, actual_label))
    return results


def show_prediction_grid(model, held_out, to_tensor):
    """Draw a grid annotating each held-out tile with predicted vs actual label.

    Uses :func:`predict_labels` for inference, then draws a matplotlib grid.
    Exactly ``len(held_out)`` tiles are annotated, each titled
    ``"pred: <predicted> / actual: <actual>"``.

    Args:
        model: The loaded model (see :func:`load_trained_model`).
        held_out: A list of ``(image, actual_label)`` pairs.
        to_tensor: A callable mapping a ``PIL.Image.Image`` to a
            ``(3, 64, 64)`` tensor.
    """
    import matplotlib  # noqa: F401  (ensure backend is initialized)
    import matplotlib.pyplot as plt

    predictions = predict_labels(model, held_out, to_tensor)

    n = len(predictions)
    cols = 4
    rows = (n + cols - 1) // cols if n else 1
    # Give each tile more width and extra row height so the two-line title has
    # room; the single-line "pred: X / actual: Y" title was wider than a 2-inch
    # tile and overlapped its neighbors.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))

    flat_axes = _flatten_axes(axes)

    for ax, (img, predicted_label, actual_label) in zip(flat_axes, predictions):
        ax.imshow(img)
        # Split the annotation across two lines and shrink the font so it fits
        # within the tile width instead of running into adjacent titles.
        ax.set_title(
            f"pred: {predicted_label}\nactual: {actual_label}",
            fontsize=9,
        )
        ax.axis("off")

    # Turn off any unused axes so empty tiles are not drawn.
    for ax in flat_axes[n:]:
        ax.axis("off")

    # Add explicit padding between subplots so titles never touch neighbors.
    plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.0)
    plt.show()


def _flatten_axes(axes):
    """Normalize a matplotlib ``subplots`` return value to a flat list of Axes.

    ``plt.subplots`` returns a single Axes for a 1x1 grid, a 1-D ndarray for a
    single row/column, and a 2-D ndarray otherwise. This coerces all of those
    to a flat Python list so callers can iterate uniformly.

    Args:
        axes: The second element returned by ``plt.subplots``.

    Returns:
        A flat list of ``matplotlib.axes.Axes``.
    """
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    return [axes]
