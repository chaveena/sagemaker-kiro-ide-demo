"""Unit tests for the failure-reporting helper in reporting.py.

Validates: Requirements 4.6 — IF the Training_Job fails, THEN the Notebook
SHALL surface the SageMaker failure reason to the User.
"""

import os
import sys

# Make the project root importable so `reporting` resolves regardless of the
# directory pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reporting import (  # noqa: E402
    NO_FAILURE_REASON,
    extract_failure_reason,
)


def test_extract_failure_reason_returns_reported_reason():
    """A present FailureReason is returned verbatim (Req 4.6)."""
    description = {"FailureReason": "AlgorithmError: something"}
    assert extract_failure_reason(description) == "AlgorithmError: something"


def test_extract_failure_reason_missing_key_returns_fallback():
    """A description with no FailureReason key yields the fallback message."""
    description = {"TrainingJobStatus": "Failed"}
    assert extract_failure_reason(description) == NO_FAILURE_REASON


def test_extract_failure_reason_none_value_returns_fallback():
    """A FailureReason of None yields the fallback message."""
    description = {"FailureReason": None}
    assert extract_failure_reason(description) == NO_FAILURE_REASON


def test_extract_failure_reason_empty_string_returns_fallback():
    """A FailureReason of "" yields the fallback message."""
    description = {"FailureReason": ""}
    assert extract_failure_reason(description) == NO_FAILURE_REASON
