"""
SQLite-backed cache for vision-language inference results.

Keyed by (image content hash, namespace), where the namespace combines model
name with a digest of every other input that influences output: base user
prompt, sampling settings, JPEG quality/dimensions. Reruns with identical
inputs skip the model call; reruns with any of those changed automatically
land in a fresh namespace instead of replaying stale results.

Persistence is a single SQLite file. SQLite handles atomicity and crash safety
for us, so the only synchronization concern is the in-process lock that guards
the shared sqlite3 Connection across worker threads.
"""

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.models import InferenceResult


if TYPE_CHECKING:
    from pathlib import Path


_HASH_DIGEST_BYTES = 16  # 128-bit BLAKE2b; collisions are not a realistic concern here.
_HASH_READ_CHUNK = 64 * 1024
_CONFIG_DIGEST_BYTES = 8  # short fingerprint, plenty for distinguishing configs.


def hash_image_file(path: Path) -> str:
    """Return a hex BLAKE2b digest of *path*'s bytes for cache lookup."""
    h = hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache_namespace(  # noqa: PLR0913 - each kwarg is a distinct input to the digest.
    model_name: str,
    *,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    jpeg_dimensions: int,
    jpeg_quality: int,
) -> str:
    """
    Combine *model_name* with a digest of every other inference input.

    The returned string is suitable as ``model_name`` for :class:`InferenceCache`.
    Two runs that match on all six arguments share cache entries; if any of them
    differs the digest changes and the cache treats them as separate namespaces.

    The system prompt is intentionally NOT folded in: it ships with the code, so
    a different system prompt means a new release, which is the right moment for
    a stale cache to be re-validated by the user anyway.
    """
    h = hashlib.blake2b(digest_size=_CONFIG_DIGEST_BYTES)
    payload = (
        f"{user_prompt}\0t={temperature}\0n={max_tokens}\0d={jpeg_dimensions}\0q={jpeg_quality}"
    )
    h.update(payload.encode("utf-8"))
    return f"{model_name}#{h.hexdigest()}"


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
        # WAL mode allows concurrent readers and avoids blocking on writes, which
        # matters when multiple worker threads hit the cache in parallel.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self._model = model_name
        self._path = db_path
        logger.debug("inference_cache_opened", file=str(db_path), model=model_name)

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

    def close(self) -> None:
        """
        Close the underlying connection. Safe to call multiple times.

        Storage errors during close are logged and swallowed so that the close
        path can never mask the run's actual exit reason.
        """
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("inference_cache_close_failed", error=str(exc))
