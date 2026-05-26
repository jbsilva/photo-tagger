"""Tests for the SQLite-backed inference cache."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from photo_tagger.cache import InferenceCache, build_cache_namespace, hash_image_file
from photo_tagger.models import InferenceResult


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


def _baseline_namespace_kwargs() -> dict[str, object]:
    """Shared keyword arguments for build_cache_namespace baseline assertions."""
    return {
        "user_prompt": "Describe the scene.",
        "temperature": 0.2,
        "max_tokens": 1200,
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
        {"max_tokens": 800},
        {"jpeg_dimensions": 2048},
        {"jpeg_quality": 95},
    ]
    seen = {baseline}
    for override in variants:
        kwargs = _baseline_namespace_kwargs() | override
        ns = build_cache_namespace("qwen/qwen3-vl-30b", **kwargs)  # type: ignore[arg-type]
        assert ns not in seen, f"namespace collision on override {override}: {ns}"
        seen.add(ns)


def test_inference_cache_namespace_isolates_configs(tmp_path: Path) -> None:
    """A hash stored under one namespace is invisible under a different one."""
    db = tmp_path / "cache.sqlite3"
    ns_a = build_cache_namespace("m", **_baseline_namespace_kwargs())  # type: ignore[arg-type]
    ns_b = build_cache_namespace(
        "m",
        **(_baseline_namespace_kwargs() | {"temperature": 0.9}),  # type: ignore[arg-type]
    )
    c_a = InferenceCache(db, model_name=ns_a)
    try:
        c_a.put("hash-1", _sample_result(title="under-a"))
    finally:
        c_a.close()

    c_b = InferenceCache(db, model_name=ns_b)
    try:
        # The same hash under a config with a different temperature must miss.
        assert c_b.get("hash-1") is None
    finally:
        c_b.close()


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
