"""Shared pytest configuration and AWS opt-in gating for the smoke tests.

The integration/smoke tests under ``tests/test_smoke_*.py`` exercise real AWS
wiring (S3 uploads and SageMaker managed training jobs). They are billable and
require credentials, so they are OFF by default and must be explicitly opted
into. This module centralizes the skip-gating logic so every AWS-touching test
uses the same rule, keeping the suite green in CI and in environments without
AWS access.

Gating rules (see helpers below):

* ``aws_smoke_enabled()`` -> True only when ``RUN_AWS_SMOKE_TESTS=1`` AND boto3
  can resolve both a region and credentials. Any AWS-touching test skips
  cleanly otherwise, with a clear reason.
* ``aws_training_smoke_enabled()`` -> additionally requires
  ``RUN_AWS_TRAINING_SMOKE=1``, because launching a managed training job spins
  up billable infrastructure and is far heavier than an S3 round-trip.

The ``slow`` marker is registered here so tests that launch real training jobs
can be marked ``@pytest.mark.slow`` without triggering unknown-marker warnings.
"""

import os

import pytest

# Environment variables that opt into the AWS-touching smoke tests.
AWS_SMOKE_ENV = "RUN_AWS_SMOKE_TESTS"
AWS_TRAINING_SMOKE_ENV = "RUN_AWS_TRAINING_SMOKE"


def pytest_configure(config):
    """Register custom markers so their use does not emit warnings."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that launch real, long-running AWS work "
        "(e.g. a managed training job).",
    )


def _boto3_can_reach_aws():
    """Return (ok, reason) describing whether boto3 can resolve AWS access.

    "ok" means boto3 imports and can resolve both a region and credentials.
    When not ok, ``reason`` explains what was missing so the skip message is
    actionable. This never makes a network call -- it only inspects locally
    resolvable configuration.
    """
    try:
        import boto3
    except ImportError:
        return False, "boto3 is not installed"

    try:
        session = boto3.session.Session()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not create a boto3 session: {exc}"

    if not session.region_name:
        return False, "no AWS region is configured"

    try:
        credentials = session.get_credentials()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not resolve AWS credentials: {exc}"

    if credentials is None:
        return False, "no AWS credentials could be resolved"

    return True, ""


def aws_smoke_reason():
    """Return a skip reason string, or None if the AWS smoke tests may run.

    The tests run only when ``RUN_AWS_SMOKE_TESTS=1`` AND boto3 can resolve a
    region and credentials.
    """
    if os.environ.get(AWS_SMOKE_ENV) != "1":
        return (
            f"AWS smoke tests are opt-in; set {AWS_SMOKE_ENV}=1 "
            "(and configure AWS credentials/region) to run them."
        )
    ok, reason = _boto3_can_reach_aws()
    if not ok:
        return f"{AWS_SMOKE_ENV}=1 is set but AWS is not reachable: {reason}."
    return None


def aws_training_smoke_reason():
    """Return a skip reason string, or None if the training smoke test may run.

    Launching a managed training job is billable and slow, so it is
    double-gated: it requires the base AWS opt-in *and* an additional
    ``RUN_AWS_TRAINING_SMOKE=1``.
    """
    base_reason = aws_smoke_reason()
    if base_reason is not None:
        return base_reason
    if os.environ.get(AWS_TRAINING_SMOKE_ENV) != "1":
        return (
            "Managed training job smoke test launches billable "
            f"infrastructure; also set {AWS_TRAINING_SMOKE_ENV}=1 to run it."
        )
    return None


def aws_smoke_enabled():
    """True when the AWS S3 smoke tests are opted into and AWS is reachable."""
    return aws_smoke_reason() is None


def aws_training_smoke_enabled():
    """True when the managed training job smoke test is fully opted into."""
    return aws_training_smoke_reason() is None
