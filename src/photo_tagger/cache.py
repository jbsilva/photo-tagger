"""
SQLite-backed cache for vision-language inference results.

Keyed by (image content hash, namespace), where the namespace combines model name with a digest of
every other input that shapes an answer: base user prompt, output language, sampling settings, JPEG
quality/dimensions. Reruns with identical inputs skip the model call; reruns with any of those
changed automatically land in a fresh namespace instead of replaying stale results. The one
inference setting left out is ``max_tokens``, which cannot shape a stored answer; see
:func:`build_cache_namespace`.

Persistence is a single SQLite file. SQLite handles atomicity and crash safety for us, so the only
synchronization concern is the in-process lock that guards the shared sqlite3 Connection across
worker threads.
"""

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Self

from loguru import logger

from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE
from photo_tagger.models import InferenceResult


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


_HASH_DIGEST_BYTES = 16  # 128-bit BLAKE2b; collisions are not a realistic concern here.
_CONFIG_DIGEST_BYTES = 8  # short fingerprint, plenty for distinguishing configs.

# Entries untouched for this long are deleted when the cache opens. Every settings change writes
# to a fresh namespace and orphans the old rows, so without pruning the file only ever grows.
CACHE_RETENTION_DAYS = 180


def hash_image_file(path: Path) -> str:
    """Return a hex BLAKE2b digest of *path*'s bytes for cache lookup."""
    with path.open("rb") as fh:
        digest = hashlib.file_digest(fh, lambda: hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES))
    return digest.hexdigest()


def build_cache_namespace(  # noqa: PLR0913 - each kwarg is a distinct input to the digest.
    model_name: str,
    *,
    user_prompt: str,
    temperature: float,
    frequency_penalty: float,
    jpeg_dimensions: int,
    jpeg_quality: int,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """
    Combine *model_name* with a digest of the inference inputs that shape an answer.

    The returned string is suitable as ``model_name`` for :class:`InferenceCache`. Two runs that
    match on every argument share cache entries; if any of them differs the digest changes and the
    cache treats them as separate namespaces.

    The system prompt text is intentionally NOT folded in: it ships with the code, so a different
    system prompt means a new release, which is the right moment for a stale cache to be re-
    validated by the user anyway. Its one runtime-configurable input, *output_language*, IS folded
    in, so switching language never replays results generated in the old one.

    ``max_tokens`` is deliberately absent, and it is the one omission worth explaining. The cap can
    only ever abort a call: an answer that reaches it is incomplete, fails validation, and never
    reaches the cache. No stored row was therefore shaped by the cap in force when it was written,
    and raising or lowering it to escape a model stuck in a repetition loop (the one reason it is
    ever reached) must not cost the user their whole cache. It was folded in once, so a cache
    written before this change answers under a different digest.
    """
    h = hashlib.blake2b(digest_size=_CONFIG_DIGEST_BYTES)
    payload = (
        f"{user_prompt}\0t={temperature}"
        f"\0fp={frequency_penalty}\0d={jpeg_dimensions}\0q={jpeg_quality}"
        f"\0l={output_language}"
    )
    h.update(payload.encode("utf-8"))
    return f"{model_name}#{h.hexdigest()}"


def content_cache_key(image_path: Path, content_hash: str | None) -> str | None:
    """
    Return the cache key for *image_path*'s pixel content, or None if it cannot be hashed.

    Prefers exiftool's ImageDataHash (*content_hash*, read alongside the image context), which
    covers the image stream only. That makes the key independent of metadata, so embedding tags does
    not change it and a later run over the same photo still hits the cache. Formats exiftool cannot
    hash that way fall back to hashing the whole file; for those, re-embedding metadata does change
    the key. A hashing failure is logged and the photo then runs without caching.
    """
    if content_hash is not None:
        return content_hash
    try:
        return hash_image_file(image_path)
    except OSError as exc:
        logger.warning("inference_cache_hash_failed", file=image_path.name, error=str(exc))
        return None


def open_cache(db_path: Path | None, *, namespace: str) -> InferenceCache | None:
    """
    Open the SQLite cache at *db_path*, or warn and degrade to no-cache on failure.

    Returns ``None`` when *db_path* is ``None`` or the cache cannot be opened. Open failures
    (permission denied, corrupt DB, unwritable parent dir) log a warning and let the run proceed
    without caching, since losing the cache should never block tagging photos.
    """
    if db_path is None:
        return None
    try:
        return InferenceCache(db_path, model_name=namespace)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("inference_cache_open_failed", file=str(db_path), error=str(exc))
        return None


def safe_cache_get(
    cache: InferenceCache,
    cache_key: str,
    *,
    file_name: str,
) -> InferenceResult | None:
    """Return the cached result for *cache_key*, treating any read error as a miss."""
    try:
        return cache.get(cache_key)
    except Exception as exc:  # noqa: BLE001 - sqlite errors must not abort the photo.
        logger.warning("inference_cache_get_failed", file=file_name, error=str(exc))
        return None


def safe_cache_put(
    cache: InferenceCache,
    cache_key: str,
    inference: InferenceResult,
    *,
    file_name: str,
) -> None:
    """Store *inference* under *cache_key*, logging and swallowing any storage error."""
    try:
        cache.put(cache_key, inference)
    except Exception as exc:  # noqa: BLE001 - sqlite errors must not abort the photo.
        logger.warning("inference_cache_put_failed", file=file_name, error=str(exc))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference (
    image_hash    TEXT NOT NULL,
    model         TEXT NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    seconds       REAL    NOT NULL DEFAULT 0.0,
    created_at    TEXT    NOT NULL,
    PRIMARY KEY (image_hash, model)
)
"""


class InferenceCache:
    """Sqlite-backed lookup table for InferenceResult by (image hash, model)."""

    __slots__ = ("_conn", "_lock", "_model", "_path")

    _conn: sqlite3.Connection
    _lock: threading.Lock
    _model: str
    _path: Path

    def __init__(self, db_path: Path, *, model_name: str) -> None:
        """Open or create the cache file at *db_path* for the given model."""
        # check_same_thread=False is safe because every read and write is wrapped
        # in the per-cache lock below; we deliberately serialize all DB access.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Every other attribute is set before the first statement runs, so nothing on this class
        # can reach a half-built instance. The prune below used to run before the lock existed,
        # which worked only because it happened not to take it.
        self._lock = threading.Lock()
        self._model = model_name
        self._path = db_path
        try:
            # WAL keeps other processes (a concurrent CLI run, the GUI) from blocking behind this
            # one's writes. Within this process the lock above serializes everything anyway.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
            self._prune_stale_rows()
        except BaseException:
            # open_cache degrades to "no cache" on failure; without this close the connection
            # (and its file handle) would leak for the life of the process.
            self._conn.close()
            raise
        logger.debug("inference_cache_opened", file=str(db_path), model=model_name)

    def _prune_stale_rows(self) -> None:
        """
        Delete entries older than :data:`CACHE_RETENTION_DAYS`, best-effort.

        ``created_at`` holds same-format UTC ISO-8601 strings, so a lexicographic comparison is a
        chronological one. A prune failure (a lock held by a concurrent run) must not disable
        caching, so it only logs.
        """
        cutoff = (datetime.now(tz=UTC) - timedelta(days=CACHE_RETENTION_DAYS)).isoformat()
        try:
            with self._conn:  # implicit transaction; commits on success
                pruned = self._conn.execute(
                    "DELETE FROM inference WHERE created_at < ?",
                    (cutoff,),
                ).rowcount
        except sqlite3.Error as exc:
            logger.warning("inference_cache_prune_failed", error=str(exc))
            return
        if pruned > 0:
            logger.info("inference_cache_pruned", rows=pruned, retention_days=CACHE_RETENTION_DAYS)

    def get(self, image_hash: str) -> InferenceResult | None:
        """Return the cached InferenceResult for *image_hash*, or None on miss."""
        with (
            self._lock,
            closing(
                self._conn.execute(
                    "SELECT title, description, keywords_json,"
                    " input_tokens, output_tokens, total_tokens, seconds"
                    " FROM inference WHERE image_hash = ? AND model = ?",
                    (image_hash, self._model),
                ),
            ) as cur,
        ):
            row = cur.fetchone()
        if row is None:
            return None
        return InferenceResult(
            title=row[0],
            description=row[1],
            keywords=json.loads(row[2]),
            input_tokens=row[3],
            output_tokens=row[4],
            total_tokens=row[5],
            seconds=row[6],
        )

    def put(self, image_hash: str, result: InferenceResult) -> None:
        """Insert or update the cache entry for *image_hash* under the active model."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO inference"
                " (image_hash, model, title, description, keywords_json,"
                "  input_tokens, output_tokens, total_tokens, seconds, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    image_hash,
                    self._model,
                    result.title,
                    result.description,
                    json.dumps(result.keywords),
                    result.input_tokens,
                    result.output_tokens,
                    result.total_tokens,
                    result.seconds,
                    datetime.now(tz=UTC).isoformat(),
                ),
            )
            self._conn.commit()

    def __enter__(self) -> Self:
        """Return the cache for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Close the cache when leaving the ``with`` block."""
        self.close()

    def close(self) -> None:
        """
        Close the underlying connection; safe to call multiple times.

        Storage errors during close are logged and swallowed so that the close path can never mask
        the run's actual exit reason.
        """
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("inference_cache_close_failed", error=str(exc))
