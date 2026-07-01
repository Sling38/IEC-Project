"""Simple on-disk cache for API responses.

UN Comtrade's free tier is aggressively rate limited and pytrends/other free
services can break unexpectedly. To keep the pipeline reproducible we cache every
raw response keyed by a hash of the request. Re-running a pull therefore hits the
network only once per unique query.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

# Repo-root/data/cache — resolved relative to this file so it works from anywhere.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


class JsonCache:
    """Namespaced key/value cache that stores JSON payloads on disk.

    Parameters
    ----------
    namespace:
        Sub-directory under the cache root, e.g. ``"comtrade"`` or ``"worldbank"``.
    cache_dir:
        Root cache directory. Defaults to ``<repo>/data/cache``.
    ttl_seconds:
        Optional freshness window. Entries older than this are treated as misses
        and re-fetched. ``None`` (default) means cached entries never expire.
    """

    def __init__(
        self,
        namespace: str,
        cache_dir: Optional[Path] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.dir = root / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        """Return the cached payload for ``key`` or ``None`` on a miss."""
        path = self._path_for(key)
        if not path.exists():
            return None
        if self.ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > self.ttl_seconds:
                return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)["payload"]
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupt cache entry — treat as a miss.
            return None

    def set(self, key: str, payload: Any) -> None:
        """Persist ``payload`` under ``key``."""
        path = self._path_for(key)
        record = {"key": key, "cached_at": time.time(), "payload": payload}
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(record, fh)
        tmp.replace(path)  # atomic write

    def clear(self) -> int:
        """Delete every entry in this namespace. Returns the number removed."""
        count = 0
        for entry in self.dir.glob("*.json"):
            entry.unlink()
            count += 1
        return count
