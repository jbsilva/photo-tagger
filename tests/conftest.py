"""Shared pytest fixtures for the photo-tagger test suite."""

from typing import TYPE_CHECKING, cast

import pytest
from pydantic_ai import BinaryContent


if TYPE_CHECKING:
    from pydantic_ai import Agent


@pytest.fixture
def stub_jpeg_bytes() -> BinaryContent:
    """Return a tiny placeholder JPEG used to short-circuit image preparation in unit tests."""
    return BinaryContent(data=b"\xff\xd8stub", media_type="image/jpeg")


@pytest.fixture
def fake_agent() -> Agent:
    """Return a typed-but-bogus Agent placeholder for tests that mock every IO collaborator."""
    return cast("Agent", object())
