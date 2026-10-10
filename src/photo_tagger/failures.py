"""
Why a photo failed, in words that carry no message text: a coarse bucket and an exception name.

Shared by the CLI pipeline and the GUI worker, so a failure reads the same in a run's summary, in
the window, and in telemetry. Classification goes by class name and module, so no provider SDK has
to be imported to name its errors.
"""

from pydantic_ai.exceptions import ModelHTTPError


# The coarse failure buckets. Classification is heuristic by exception class name/module so no
# provider SDK needs importing here; the buckets answer "why do photos fail" without carrying
# any message text.
FAILURE_TIMEOUT = "timeout"
FAILURE_CONNECTION = "connection"
FAILURE_MODEL_VALIDATION = "model-validation"
FAILURE_MODEL_API = "model-api"
FAILURE_IMAGE_READ = "image-read"
FAILURE_METADATA_WRITE = "metadata-write"
FAILURE_OTHER = "other"


def classify_failure(exc: BaseException) -> str:
    """Map an exception from one photo's processing to a coarse failure bucket."""
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return FAILURE_TIMEOUT
    if "Connect" in name or "Network" in name or "Pool" in name:
        return FAILURE_CONNECTION
    if "UnexpectedModelBehavior" in name or "Validation" in name:
        return FAILURE_MODEL_VALIDATION
    if module.startswith(("rawpy", "PIL")) or "Image" in name:
        return FAILURE_IMAGE_READ
    if "Status" in name or "HTTP" in name or "API" in name:
        return FAILURE_MODEL_API
    return FAILURE_OTHER


def failure_type(exc: BaseException) -> str:
    """
    Name the exception behind a failed photo, with the HTTP status if it has one.

    ``ModelHTTPError(404)`` tells a missing model from a rejected key, where the bucket only says
    "model-api". A class name and a status code are facts about the code and the server, never text
    from the photo or the server's reply.
    """
    name = type(exc).__name__
    status = _status_code(exc)
    return f"{name}({status})" if status is not None else name


def _status_code(exc: BaseException) -> int | None:
    """Return the HTTP status an SDK exception carries, on itself or on its response."""
    for holder in (exc, _attribute(exc, "response")):
        status = _attribute(holder, "status_code")
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def _attribute(value: object, name: str) -> object:
    """Read an attribute that an SDK may implement as a property that raises."""
    try:
        return getattr(value, name, None)
    except Exception:  # noqa: BLE001 - an SDK property may raise anything; there is no status then.
        return None


# HTTP status codes that retrying cannot fix: the request itself is rejected, not the server
# having a bad moment. Deliberately narrow (contrast 429/500/502/503, which a 5s pause and a
# second attempt can plausibly clear): a wrong or revoked credential does not become right by
# waiting, but almost every other status code pydantic-ai's ModelHTTPError can carry might.
_NON_RETRYABLE_STATUS_CODES = frozenset({401, 403})


def is_permanent_failure(exc: BaseException) -> bool:
    """Report whether *exc* is one no retry pass could ever recover from."""
    return isinstance(exc, ModelHTTPError) and exc.status_code in _NON_RETRYABLE_STATUS_CODES
