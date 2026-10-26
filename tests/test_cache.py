"""Tests for the SQLite-backed inference cache."""

import inspect
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from photo_tagger.cache import (
    InferenceCache,
    build_cache_namespace,
    content_cache_key,
    hash_image_file,
    open_cache,
    safe_cache_get,
    safe_cache_put,
)
from photo_tagger.config import DEFAULT_USER_PROMPT
from photo_tagger.models import InferenceResult


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def _sample_result(*, title: str = "T", description: str = "D") -> InferenceResult:
    """Return a small InferenceResult that round-trips through the cache."""
    return InferenceResult(
        title=title,
        description=description,
        keywords=["Beach", "Sunset"],
        input_tokens=42,
        output_tokens=7,
        total_tokens=49,
        seconds=0.5,
    )


def test_hash_image_file_is_stable_for_same_contents(tmp_path: Path) -> None:
    """Identical bytes produce identical hashes and a tweak changes them."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"\xff\xd8hello world")
    b.write_bytes(b"\xff\xd8hello world")
    assert hash_image_file(a) == hash_image_file(b)

    b.write_bytes(b"\xff\xd8hello-world")  # single-byte tweak: " " becomes "-"
    assert hash_image_file(a) != hash_image_file(b)


def test_content_cache_key_prefers_the_content_hash(tmp_path: Path) -> None:
    """When exiftool supplied an ImageDataHash, it is the key and the file is never read."""
    assert content_cache_key(tmp_path / "ghost.jpg", "abc123") == "abc123"


def test_content_cache_key_falls_back_to_the_whole_file_hash(tmp_path: Path) -> None:
    """Without a content hash the key degrades to hashing the file bytes."""
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8data")
    assert content_cache_key(img, None) == hash_image_file(img)


def test_content_cache_key_returns_none_when_hashing_fails(tmp_path: Path) -> None:
    """An unreadable file yields None (photo runs uncached), never an exception."""
    assert content_cache_key(tmp_path / "ghost.jpg", None) is None


def test_open_cache_returns_none_when_path_is_none() -> None:
    """No cache file configured means no cache, never an exception."""
    assert open_cache(None, namespace="m#x") is None


def test_open_cache_degrades_on_open_failure(tmp_path: Path) -> None:
    """An unusable cache target is logged and downgraded to no-cache, not raised."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache's parent dir should be")
    assert open_cache(blocker / "cache.sqlite3", namespace="m#x") is None


def test_open_cache_returns_a_working_cache(tmp_path: Path) -> None:
    """
    The happy path yields a real, usable cache.

    Only the degradation cases were asserted before, all of which expect None; a regression that
    made open_cache always degrade would have run every install uncached with a green suite.
    """
    cache = open_cache(tmp_path / "cache.sqlite3", namespace="m#x")
    assert cache is not None
    with cache:
        cache.put("hash-1", _sample_result(title="Round trip"))
        got = cache.get("hash-1")
        assert got is not None
        assert got.title == "Round trip"


def test_cache_prunes_stale_rows_on_open(tmp_path: Path) -> None:
    """
    Rows older than the retention window are deleted when the cache opens; fresh ones stay.

    Every settings change writes to a fresh namespace and orphans the old rows, so without this the
    cache file only ever grows.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415 - test-local

    from photo_tagger.cache import CACHE_RETENTION_DAYS, InferenceCache  # noqa: PLC0415

    db = tmp_path / "cache.sqlite3"
    with InferenceCache(db, model_name="m#x") as cache:
        cache.put("fresh", InferenceResult(title="T", description="D", keywords=[]))
        stale_stamp = (datetime.now(tz=UTC) - timedelta(days=CACHE_RETENTION_DAYS + 1)).isoformat()
        cache._conn.execute(  # noqa: SLF001 - plant an old row directly
            "UPDATE inference SET created_at = ? WHERE image_hash = 'fresh'",
            (stale_stamp,),
        )
        cache._conn.execute(  # noqa: SLF001
            "INSERT INTO inference SELECT 'stale', model, title, description, keywords_json,"
            " input_tokens, output_tokens, total_tokens, seconds, created_at"
            " FROM inference WHERE image_hash = 'fresh'",
        )
        cache._conn.execute(  # noqa: SLF001 - restore the fresh row's stamp
            "UPDATE inference SET created_at = ? WHERE image_hash = 'fresh'",
            (datetime.now(tz=UTC).isoformat(),),
        )
        cache._conn.commit()  # noqa: SLF001

    with InferenceCache(db, model_name="m#x") as reopened:
        assert reopened.get("stale") is None
        assert reopened.get("fresh") is not None


def test_open_cache_closes_connection_when_setup_fails(tmp_path: Path) -> None:
    """
    A corrupt DB file degrades to no-cache AND closes the already-opened connection.

    Regression test: the connection (and its file handle) used to leak for the life of the process
    when the PRAGMA/schema setup raised after sqlite3.connect succeeded.
    """
    corrupt = tmp_path / "cache.sqlite3"
    corrupt.write_bytes(b"this is not a sqlite database, it just lives at the cache path")

    opened: list[MagicMock] = []
    real_connect = sqlite3.connect

    def tracking_connect(database: str, *, check_same_thread: bool = True) -> MagicMock:
        conn = real_connect(database, check_same_thread=check_same_thread)
        wrapper = MagicMock(wraps=conn)
        opened.append(wrapper)
        return wrapper

    with patch("photo_tagger.cache.sqlite3.connect", side_effect=tracking_connect):
        assert open_cache(corrupt, namespace="m#x") is None

    assert len(opened) == 1
    opened[0].close.assert_called_once()


def test_cache_is_fully_built_before_the_first_statement_runs(tmp_path: Path) -> None:
    """
    Every attribute exists by the time the prune (or any other DB work) can touch it.

    The prune used to run before ``_lock`` was assigned, and worked only because it happened not to
    take the lock: the class uses ``__slots__``, so reaching for it there raised AttributeError.
    Assert the invariant rather than the ordering, so the next statement added to __init__ is
    covered too.
    """
    seen: list[str] = []

    def record(self: InferenceCache) -> None:
        seen.extend(name for name in InferenceCache.__slots__ if hasattr(self, name))

    with patch.object(InferenceCache, "_prune_stale_rows", record):
        InferenceCache(tmp_path / "cache.sqlite3", model_name="m#x").close()

    assert sorted(seen) == sorted(InferenceCache.__slots__)


def test_safe_cache_get_treats_errors_as_misses() -> None:
    """A read error inside sqlite is swallowed and reported as a miss."""
    broken = MagicMock()
    broken.get.side_effect = sqlite3.OperationalError("locked")
    assert safe_cache_get(broken, "key", file_name="img.jpg") is None


def test_safe_cache_put_swallows_errors() -> None:
    """A write error inside sqlite is swallowed, never raised at the caller."""
    broken = MagicMock()
    broken.put.side_effect = sqlite3.OperationalError("disk full")
    safe_cache_put(broken, "key", _sample_result(), file_name="img.jpg")  # must not raise


@pytest.fixture
def make_cache() -> Iterator[Callable[..., InferenceCache]]:
    """
    Build InferenceCache instances that always close when the test ends.

    Hand-rolled try/finally around every cache was easy to forget, and a leaked WAL-mode SQLite
    handle bleeds into later tests on the same xdist worker.
    """
    with ExitStack() as stack:

        def _make(db: Path, *, model_name: str) -> InferenceCache:
            return stack.enter_context(InferenceCache(db, model_name=model_name))

        yield _make


def test_inference_cache_round_trip(
    tmp_path: Path,
    make_cache: Callable[..., InferenceCache],
) -> None:
    """A put/get round-trip returns equal InferenceResult fields."""
    cache = make_cache(tmp_path / "cache.sqlite3", model_name="m1")
    cache.put("hash-1", _sample_result(title="Forest"))
    got = cache.get("hash-1")
    assert got is not None
    assert got.title == "Forest"
    assert got.keywords == ["Beach", "Sunset"]
    assert got.total_tokens == _sample_result().total_tokens


def test_inference_cache_get_returns_none_on_miss(
    tmp_path: Path,
    make_cache: Callable[..., InferenceCache],
) -> None:
    """A lookup for an unknown hash returns None instead of raising."""
    cache = make_cache(tmp_path / "cache.sqlite3", model_name="m1")
    assert cache.get("never-stored") is None


def test_inference_cache_keys_by_model_name(
    tmp_path: Path,
    make_cache: Callable[..., InferenceCache],
) -> None:
    """The same hash under a different model name is a cache miss."""
    db = tmp_path / "cache.sqlite3"
    make_cache(db, model_name="model-a").put("hash-1", _sample_result())
    assert make_cache(db, model_name="model-b").get("hash-1") is None


def test_inference_cache_replaces_existing_entry(
    tmp_path: Path,
    make_cache: Callable[..., InferenceCache],
) -> None:
    """A second put for the same (hash, model) overwrites the prior row."""
    cache = make_cache(tmp_path / "cache.sqlite3", model_name="m1")
    cache.put("hash-1", _sample_result(title="Old"))
    cache.put("hash-1", _sample_result(title="New"))
    got = cache.get("hash-1")
    assert got is not None
    assert got.title == "New"


def _baseline_namespace_kwargs() -> dict[str, object]:
    """Shared keyword arguments for build_cache_namespace baseline assertions."""
    return {
        "user_prompt": "Describe the scene.",
        "temperature": 0.2,
        "frequency_penalty": 0.5,
        "jpeg_dimensions": 1280,
        "jpeg_quality": 80,
    }


def test_build_cache_namespace_is_stable_for_same_inputs() -> None:
    """Two calls with identical arguments return the same namespace string."""
    kwargs = _baseline_namespace_kwargs()
    a = build_cache_namespace("qwen/qwen3-vl-30b", **kwargs)  # type: ignore[arg-type]
    b = build_cache_namespace("qwen/qwen3-vl-30b", **kwargs)  # type: ignore[arg-type]
    assert a == b
    assert a.startswith("qwen/qwen3-vl-30b#")


def test_build_cache_namespace_changes_with_each_input() -> None:
    """Changing any single argument produces a distinct namespace."""
    baseline = build_cache_namespace(
        "qwen/qwen3-vl-30b",
        **_baseline_namespace_kwargs(),  # type: ignore[arg-type]
    )

    variants = [
        {"user_prompt": "Different prompt."},
        {"temperature": 0.9},
        {"frequency_penalty": 1.0},
        {"jpeg_dimensions": 2048},
        {"jpeg_quality": 95},
        {"output_language": "German"},
    ]
    seen = {baseline}
    for override in variants:
        kwargs = _baseline_namespace_kwargs() | override
        ns = build_cache_namespace("qwen/qwen3-vl-30b", **kwargs)  # type: ignore[arg-type]
        assert ns not in seen, f"namespace collision on override {override}: {ns}"
        seen.add(ns)


def test_inference_cache_namespace_isolates_configs(
    tmp_path: Path,
    make_cache: Callable[..., InferenceCache],
) -> None:
    """A hash stored under one namespace is invisible under a different one."""
    db = tmp_path / "cache.sqlite3"
    ns_a = build_cache_namespace("m", **_baseline_namespace_kwargs())  # type: ignore[arg-type]
    ns_b = build_cache_namespace(
        "m",
        **(_baseline_namespace_kwargs() | {"temperature": 0.9}),  # type: ignore[arg-type]
    )
    make_cache(db, model_name=ns_a).put("hash-1", _sample_result(title="under-a"))
    # The same hash under a config with a different temperature must miss.
    assert make_cache(db, model_name=ns_b).get("hash-1") is None


def test_inference_cache_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent puts under different keys all land without raising or losing entries."""
    cache = InferenceCache(tmp_path / "cache.sqlite3", model_name="m1")
    try:
        keys = [f"hash-{i:03d}" for i in range(80)]

        def put_one(key: str) -> None:
            cache.put(key, _sample_result(title=key))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(put_one, keys))

        for key in keys:
            got = cache.get(key)
            assert got is not None
            assert got.title == key
    finally:
        cache.close()


def test_inference_cache_close_swallows_sqlite_error(tmp_path: Path) -> None:
    """A storage error raised while closing is logged and never propagated."""
    cache = InferenceCache(tmp_path / "cache.sqlite3", model_name="m1")
    cache._conn.close()  # noqa: SLF001 - release the real handle before swapping in the stub

    failing = MagicMock()
    failing.close.side_effect = sqlite3.Error("already closed")
    cache._conn = failing  # noqa: SLF001 - force the guarded close path to fail

    cache.close()  # must not raise
    failing.close.assert_called_once()


def test_inference_cache_context_manager_round_trip(tmp_path: Path) -> None:
    """Using InferenceCache as a context manager auto-closes the connection."""
    with InferenceCache(tmp_path / "cache.sqlite3", model_name="m1") as cache:
        cache.put("hash-1", _sample_result(title="CM"))
        got = cache.get("hash-1")
        assert got is not None
        assert got.title == "CM"
    # After __exit__, a new cache on the same file must succeed (old conn is closed).
    with InferenceCache(tmp_path / "cache.sqlite3", model_name="m1") as cache:
        assert cache.get("hash-1") is not None


def test_inference_cache_context_manager_closes_on_exception(tmp_path: Path) -> None:
    """The cache connection is closed even when the with-block raises."""
    db = tmp_path / "cache.sqlite3"

    def _use_and_crash(cache: InferenceCache) -> None:
        cache.put("hash-1", _sample_result())
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"), InferenceCache(db, model_name="m1") as cache:
        _use_and_crash(cache)

    # Re-opening must succeed; the old connection was released.
    with InferenceCache(db, model_name="m1") as cache:
        got = cache.get("hash-1")
        assert got is not None


# The digest every install's cache is filed under. Pinned so a change to the payload format, or to
# one of the inputs folded into it, cannot silently orphan every user's cached results: a release
# that means to do that has to edit this number and say so in the changelog.
_PINNED_NAMESPACE = "qwen/qwen3-vl-30b#ab84e0fd8273877d"


def test_build_cache_namespace_digest_is_pinned() -> None:
    """The namespace for a known set of inputs never moves without someone deciding it should."""
    assert (
        build_cache_namespace(
            "qwen/qwen3-vl-30b",
            user_prompt=DEFAULT_USER_PROMPT,
            temperature=0.2,
            frequency_penalty=0.5,
            jpeg_dimensions=640,
            jpeg_quality=80,
            output_language="English",
        )
        == _PINNED_NAMESPACE
    )


def test_build_cache_namespace_ignores_the_token_cap() -> None:
    """
    The cap is not an input: it cannot shape an answer that was stored.

    A call that reaches it returns nothing usable, so raising the cap to escape a model stuck in a
    repetition loop must leave every cached result readable.
    """
    assert "max_tokens" not in inspect.signature(build_cache_namespace).parameters
