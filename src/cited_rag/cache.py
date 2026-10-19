"""On-disk embedding cache keyed by a content hash.

Key = sha256(model name, vector kind, text). Values are float32 arrays stored in SQLite,
so the cache is one file, safe to share between runs, and survives re-chunking: any chunk
whose text did not change is a cache hit.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np


def content_key(model: str, kind: str, text: str) -> str:
    h = hashlib.sha256()
    for part in (model, kind, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, shape TEXT, vec BLOB)"
        )
        self.hits = 0
        self.misses = 0

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            q = f"SELECT key, shape, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            for key, shape, blob in self._db.execute(q, batch):
                dims = tuple(int(x) for x in shape.split(","))
                out[key] = np.frombuffer(blob, dtype=np.float32).reshape(dims)
        self.hits += len(out)
        self.misses += len(set(keys) - out.keys())
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        rows = [
            (k, ",".join(map(str, v.shape)), np.ascontiguousarray(v, dtype=np.float32).tobytes())
            for k, v in items.items()
        ]
        self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
        self._db.commit()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])

    def close(self) -> None:
        self._db.close()
