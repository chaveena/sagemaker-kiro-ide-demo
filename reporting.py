"""Pure, testable result/failure reporting helpers for the satellite demo.

The guided notebook launches a SageMaker managed training job and then reports
the outcome: on success, where the model artifact landed; on failure, the
reason SageMaker gave. The reason-extraction logic is factored out here so it
is directly unit-testable without any AWS access.

Design constraint:
    Importing this module must NOT require ``boto3`` or ``sagemaker``. The
    notebook still performs the ``describe_training_job`` call (which needs
    AWS) and passes the resulting description dict into these pure functions.
"""

# Fallback used when a failed job description carries no usable failure reason.
NO_FAILURE_REASON = "No failure reason reported by SageMaker."


def extract_failure_reason(job_description):
    """Return the SageMaker failure reason from a training-job description.

    Args:
        job_description: A training-job description dict, shaped like the value
            returned by boto3's ``sagemaker.describe_training_job`` (which
            includes a ``"FailureReason"`` key when a job fails).

    Returns:
        The ``FailureReason`` string when present and non-empty. If the key is
        missing, ``None``, or an empty string, a clear fallback message
        (:data:`NO_FAILURE_REASON`) is returned instead so the user always sees
        an explanation rather than a bare ``None``.
    """
    reason = job_description.get("FailureReason")
    if reason:
        return reason
    return NO_FAILURE_REASON


def format_training_result(model_data):
    """Return a human-readable success line for the trained model artifact.

    Mirrors the success path of the notebook's result reporting (Requirement
    4.5), where the S3 URI of the produced ``model.tar.gz`` is surfaced.

    Args:
        model_data: The S3 URI of the produced model artifact (typically
            ``estimator.model_data``).

    Returns:
        A short string identifying the model artifact location.
    """
    return f"Model artifact: {model_data}"
