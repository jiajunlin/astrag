"""Layer 1 — incremental parsing cache (stdlib-only: sqlite3 + hashlib).

Full re-parsing is wasteful once a repo is large: most files are
unchanged between runs. ``IndexCache`` stores, per file, a fast
``(mtime, size)`` fingerprint plus a content hash and the serialized
``CodeChunk`` list that file produced. ``index_repo(..., cache_path=...)``
uses it to skip re-parsing any file whose fingerprint still matches.

Design:

* The fast path never hashes the file: if ``mtime`` and ``size`` on disk
  match what's stored, the cached chunks are reused with no I/O beyond
  ``os.stat``. This is the common case (nothing changed since last run).
* If ``mtime``/``size`` differ (edited, or checked out from git with a
  fresh mtime), the file is read once and its content hash is compared
  against the stored hash — a real edit re-parses; a no-op touch (same
  bytes, new mtime) still hits cache and just refreshes the fingerprint.
* Deleted files are dropped from the cache at the end of a full
  ``index_repo`` pass so a saved index never resurrects removed code.

This trades zero extra dependencies for a bit of manual (de)serialization
of ``CodeChunk`` — it's a flat dataclass of JSON-safe fields, so
``dataclasses.asdict`` / ``CodeChunk(**d)`` round-trip cleanly.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict

from .parsing import CodeChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path      TEXT PRIMARY KEY,
    mtime     REAL NOT NULL,
    size      INTEGER NOT NULL,
    hash      TEXT NOT NULL,
    chunks    TEXT NOT NULL
);
"""


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


class IndexCache:
    """Sqlite-backed cache: rel_path -> (mtime, size, hash, [CodeChunk])."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._con = sqlite3.connect(path)
        self._con.execute(_SCHEMA)
        self._con.commit()
        self._seen: set[str] = set()   # paths touched this run, for prune()

    # ---- lookups ----
    def lookup(self, rel_path: str, mtime: float, size: int,
              read_text) -> list[CodeChunk] | None:
        """Return cached chunks for ``rel_path`` if still fresh.

        ``read_text`` is a zero-arg callable that lazily reads the file's
        current content — only invoked if the fast (mtime, size) check
        misses, so unchanged files never pay for a re-read.
        """
        self._seen.add(rel_path)
        row = self._con.execute(
            "SELECT mtime, size, hash, chunks FROM files WHERE path = ?",
            (rel_path,)).fetchone()
        if row is None:
            return None
        old_mtime, old_size, old_hash, chunks_json = row
        if old_mtime == mtime and old_size == size:
            return self._decode(chunks_json)
        # fingerprint changed -> only a real content diff invalidates
        text = read_text()
        if content_hash(text) == old_hash:
            self._con.execute(
                "UPDATE files SET mtime = ?, size = ? WHERE path = ?",
                (mtime, size, rel_path))
            self._con.commit()
            return self._decode(chunks_json)
        return None

    def store(self, rel_path: str, mtime: float, size: int, text: str,
             chunks: list[CodeChunk]) -> None:
        self._seen.add(rel_path)
        payload = json.dumps([asdict(c) for c in chunks])
        self._con.execute(
            "INSERT INTO files (path, mtime, size, hash, chunks) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "mtime=excluded.mtime, size=excluded.size, "
            "hash=excluded.hash, chunks=excluded.chunks",
            (rel_path, mtime, size, content_hash(text), payload))
        self._con.commit()

    def prune(self) -> int:
        """Drop cache rows for files not touched this run (i.e. deleted).

        Call once after a full repo walk. Returns the number of rows removed.
        """
        rows = [r[0] for r in
               self._con.execute("SELECT path FROM files").fetchall()]
        stale = [p for p in rows if p not in self._seen]
        if stale:
            self._con.executemany("DELETE FROM files WHERE path = ?",
                                  [(p,) for p in stale])
            self._con.commit()
        return len(stale)

    def close(self) -> None:
        self._con.close()

    @staticmethod
    def _decode(chunks_json: str) -> list[CodeChunk]:
        return [CodeChunk(**d) for d in json.loads(chunks_json)]

    def __enter__(self) -> "IndexCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
