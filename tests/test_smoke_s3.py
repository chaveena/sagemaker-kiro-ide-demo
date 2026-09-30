"""Integration/smoke test: staging the subset in S3 (Task 8.1).

Validates: Requirement 2.4 -- the prepared ImageFolder-style subset is uploaded
to the SageMaker default bucket under the demo prefix so it can serve as the
managed training job's input channel.

This test exercises real AWS (S3) and is therefore OFF by default. It runs only
when ``RUN_AWS_SMOKE_TESTS=1`` is set AND boto3 can resolve a region and
credentials; otherwise it skips cleanly with a clear reason (see
``tests/conftest.py``). When enabled it mirrors the notebook's boto3-based
upload approach: it walks a tiny local ImageFolder-style directory, uploads
each file under a unique test prefix, asserts the objects exist at the expected
keys, then removes the uploaded objects and the local temp directory.
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

from conftest import aws_smoke_enabled, aws_smoke_reason  # noqa: E402

pytestmark = pytest.mark.skipif(
    not aws_smoke_enabled(), reason=aws_smoke_reason() or "AWS smoke disabled"
)


def _make_tiny_imagefolder(root):
    """Create a tiny ImageFolder-style dir with two classes of small images.

    Returns a dict mapping each relative ``<class>/<file>`` path to its local
    absolute path so the test can assert on expected S3 keys.
    """
    layout = {"forest": 2, "water": 2}
    relpath_to_local = {}
    for class_name, count in layout.items():
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for index in range(count):
            # Small solid-color RGB tile; content is irrelevant for an S3 test.
            img = Image.new("RGB", (16, 16), color=(index * 10, 100, 150))
            file_name = f"{index}.png"
            local_path = os.path.join(class_dir, file_name)
            img.save(local_path)
            relpath_to_local[f"{class_name}/{file_name}"] = local_path
    return relpath_to_local


def test_subset_upload_objects_exist_at_expected_prefix():
    """Uploaded subset objects exist at the expected S3 keys (Req 2.4).

    Mirrors the notebook: walk a local ImageFolder-style directory and upload
    each file with boto3 under ``s3://<default-bucket>/<unique-test-prefix>/``,
    preserving the ``<class>/<file>`` layout. Then confirm every object is
    present via ``head_object`` and that ``list_objects_v2`` returns exactly the
    expected keys.
    """
    import boto3
    from sagemaker.core.helper.session_helper import Session

    bucket = Session().default_bucket()
    s3 = boto3.client("s3")

    # Unique prefix per run so concurrent/re-run tests never collide.
    key_prefix = f"satellite-land-cover-demo-smoke/{uuid.uuid4().hex}/data"
    local_dir = tempfile.mkdtemp(prefix="s3_smoke_")
    uploaded_keys = []

    try:
        relpath_to_local = _make_tiny_imagefolder(local_dir)

        # Upload, mirroring the notebook's os.walk + upload_file approach.
        for root, _dirs, files in os.walk(local_dir):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{key_prefix}/" + rel_path.replace(os.sep, "/")
                s3.upload_file(local_path, bucket, s3_key)
                uploaded_keys.append(s3_key)

        expected_keys = {
            f"{key_prefix}/{rel}" for rel in relpath_to_local
        }

        # Each expected object must exist (head_object raises if it does not).
        for key in expected_keys:
            head = s3.head_object(Bucket=bucket, Key=key)
            assert head["ResponseMetadata"]["HTTPStatusCode"] == 200

        # Listing the prefix must return exactly the uploaded objects.
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"{key_prefix}/")
        listed_keys = {obj["Key"] for obj in listing.get("Contents", [])}
        assert listed_keys == expected_keys
    finally:
        # Best-effort cleanup of uploaded objects and the local temp dir.
        for key in uploaded_keys:
            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except Exception:
                pass
        shutil.rmtree(local_dir, ignore_errors=True)
