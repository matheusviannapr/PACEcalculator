"""Cache em disco simples, com TTL e chaveamento por hash do payload."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class DiskCache:
    """
    Cache JSON em disco. Serve tanto para respostas do Overpass/Nominatim
    quanto para perfis do PVWatts.

    ``ttl_s=None`` significa que a entrada nunca expira (geometria de
    edificação muda pouco; dados climáticos, nunca).
    """

    def __init__(self, directory: str | os.PathLike[str], ttl_s: float | None = None) -> None:
        self.directory = Path(directory)
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, payload: Any) -> str:
        raw = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, payload: Any) -> Any | None:
        path = self._path(self.key(payload))
        if not path.exists():
            return None
        if self.ttl_s is not None and (time.time() - path.stat().st_mtime) > self.ttl_s:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            # entrada corrompida (ex.: escrita interrompida): trata como ausente
            return None

    def set(self, payload: Any, value: Any) -> None:
        path = self._path(self.key(payload))
        # escrita atômica: evita cache corrompido se o processo morrer no meio
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(self.directory), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    def clear(self) -> int:
        removed = 0
        for item in self.directory.glob("*.json"):
            item.unlink(missing_ok=True)
            removed += 1
        return removed
