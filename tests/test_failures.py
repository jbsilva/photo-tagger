"""Tests for how a failed photo is named: a coarse bucket, an exception name, and permanence."""

from typing import cast

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from photo_tagger.failures import classify_failure, failure_type, is_permanent_failure


def _exception_like(name: str, module: str = "builtins") -> BaseException:
    """Build an exception instance whose class name/module mirror a real SDK's."""
    cls = cast("type[Exception]", type(name, (Exception,), {"__module__": module}))
    return cls()


@pytest.mark.parametrize(
    ("exc", "bucket"),
    [
        (TimeoutError(), "timeout"),
        (_exception_like("ReadTimeout", "httpx2"), "timeout"),
        (_exception_like("ConnectError", "httpx2"), "connection"),
        (_exception_like("UnexpectedModelBehavior", "pydantic_ai"), "model-validation"),
        (_exception_like("ValidationError", "pydantic"), "model-validation"),
        (_exception_like("HTTPStatusError", "httpx2"), "model-api"),
        (_exception_like("LibRawFileUnsupportedError", "rawpy._rawpy"), "image-read"),
        (_exception_like("UnidentifiedImageError", "PIL"), "image-read"),
        (RuntimeError("anything"), "other"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_classify_failure_buckets_common_exceptions(exc: BaseException, bucket: str) -> None:
    """Exception classes map to the coarse buckets by name/module, never by message."""
    assert classify_failure(exc) == bucket


class _Response:
    status_code = 503


class _BrokenResponseError(Exception):
    @property
    def response(self) -> object:
        msg = "no response yet"
        raise RuntimeError(msg)


def test_failure_type_adds_the_http_status_when_there_is_one() -> None:
    """The status comes from the exception or its response; an odd SDK property is survived."""
    assert failure_type(ModelHTTPError(status_code=404, model_name="m")) == "ModelHTTPError(404)"
    on_response = _exception_like("HTTPStatusError", "httpx2")
    on_response.response = _Response()  # type: ignore[attr-defined]
    assert failure_type(on_response) == "HTTPStatusError(503)"
    assert failure_type(_BrokenResponseError()) == "_BrokenResponseError"
    flagged = _exception_like("Weird")
    flagged.status_code = True  # type: ignore[attr-defined]
    assert failure_type(flagged) == "Weird"


def test_only_a_rejected_credential_is_permanent() -> None:
    """A 401 or 403 does not become right by waiting; a 503 might."""
    assert is_permanent_failure(ModelHTTPError(status_code=401, model_name="m"))
    assert is_permanent_failure(ModelHTTPError(status_code=403, model_name="m"))
    assert not is_permanent_failure(ModelHTTPError(status_code=503, model_name="m"))
    assert not is_permanent_failure(TimeoutError())
