"""Tests for the SQLite-backed inference cache."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from photo_tagger.ai import InferenceResult
from photo_tagger.cache import InferenceCache, hash_image_file


if TYPE_CHECKING:
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


def test_inference_cache_round_trip(tmp_path: Path) -> None:
    """A put/get round-trip returns equal InferenceResult fields."""
    cache = InferenceCache(tmp_path / "cache.sqlite3", model_name="m1")
    try:
        cache.put("hash-1", _sample_result(title="Forest"))
        got = cache.get("hash-1")
        assert got is not None
        assert got.title == "Forest"
        assert got.keywords == ["Beach", "Sunset"]
        assert got.total_tokens == _sample_result().total_tokens
    finally:
        cache.close()


def test_inference_cache_get_returns_none_on_miss(tmp_path: Path) -> None:
    """A lookup for an unknown hash returns None instead of raising."""
    cache = InferenceCache(tmp_path / "cache.sqlite3", model_name="m1")
    try:
        assert cache.get("never-stored") is None
    finally:
        cache.close()


def test_inference_cache_keys_by_model_name(tmp_path: Path) -> None:
    """The same hash under a different model name is a cache miss."""
    db = tmp_path / "cache.sqlite3"
    c_a = InferenceCache(db, model_name="model-a")
    try:
        c_a.put("hash-1", _sample_result())
    finally:
        c_a.close()

    c_b = InferenceCache(db, model_name="model-b")
    try:
        assert c_b.get("hash-1") is None
    finally:
        c_b.close()


def test_inference_cache_replaces_existing_entry(tmp_path: Path) -> None:
    """A second put for the same (hash, model) overwrites the prior row."""
    cache = InferenceCache(tmp_path / "cache.sqlite3", model_name="m1")
    try:
        cache.put("hash-1", _sample_result(title="Old"))
        cache.put("hash-1", _sample_result(title="New"))
        got = cache.get("hash-1")
        assert got is not None
        assert got.title == "New"
    finally:
        cache.close()


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
