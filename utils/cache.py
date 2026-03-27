"""
utils/cache.py
~~~~~~~~~~~~~~
Content-addressable on-disk cache for :class:`~core.SimulationResult` objects.

Each entry is keyed by a deterministic MD5 hash of the rounded parameter
values.  Results are serialised with ``pickle`` and stored as individual
files in *cache_dir*.

Thread-safety is provided by a :class:`threading.Lock`.

Usage example::

    cache = SimulationCache(cache_dir="results/cache", enabled=True)

    # Try to retrieve a cached result
    result = cache.get({"R1": 10e3, "C1": 1e-8})
    if result is None:
        result = run_expensive_simulation(...)
        cache.put({"R1": 10e3, "C1": 1e-8}, result)

    print(cache.stats())
    # {'hits': 1, 'misses': 1, 'size': 1}
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, Optional

from core import ParameterSet, SimulationResult

logger = logging.getLogger(__name__)


class SimulationCache:
    """Content-addressable disk cache for :class:`~core.SimulationResult`.

    Parameters
    ----------
    cache_dir:
        Directory where cached files are stored.  Created automatically
        if it does not exist.
    enabled:
        If ``False``, all operations are no-ops and the cache is
        effectively disabled.  Useful for debugging.
    hash_precision:
        Number of decimal places to round each parameter value before
        hashing.  Higher values mean more cache misses for near-identical
        parameters; lower values risk false cache hits.  Default: ``6``.
    """

    def __init__(
        self,
        cache_dir: str,
        enabled: bool = True,
        hash_precision: int = 6,
    ) -> None:
        self.cache_dir = str(cache_dir)
        self.enabled = enabled
        self.hash_precision = hash_precision

        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        if enabled:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            logger.debug(
                "SimulationCache ready: dir=%s  precision=%d",
                self.cache_dir,
                hash_precision,
            )
        else:
            logger.debug("SimulationCache disabled")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, params: ParameterSet) -> Optional[SimulationResult]:
        """Return the cached result for *params*, or ``None`` on a miss.

        Parameters
        ----------
        params:
            Parameter set to look up.

        Returns
        -------
        SimulationResult or None
        """
        if not self.enabled:
            return None

        key = self._make_key(params)
        cache_file = self._key_to_path(key)

        with self._lock:
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as fh:
                        result: SimulationResult = pickle.load(fh)
                    self._hits += 1
                    logger.debug("Cache HIT  key=%s  params=%s", key[:8], params)
                    return result
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Cache read error for key %s: %s – treating as miss",
                        key[:8],
                        exc,
                    )
            self._misses += 1
            logger.debug("Cache MISS key=%s  params=%s", key[:8], params)
            return None

    def put(self, params: ParameterSet, result: SimulationResult) -> None:
        """Store *result* in the cache under the hash of *params*.

        Parameters
        ----------
        params:
            Parameter set used to generate *result*.
        result:
            Simulation result to cache.
        """
        if not self.enabled:
            return

        key = self._make_key(params)
        cache_file = self._key_to_path(key)

        with self._lock:
            try:
                with open(cache_file, "wb") as fh:
                    pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(
                    "Cache PUT  key=%s  file=%s", key[:8], cache_file.name
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Cache write error for key %s: %s", key[:8], exc)

    def exists(self, params: ParameterSet) -> bool:
        """Return ``True`` if a cache entry for *params* exists.

        Parameters
        ----------
        params:
            Parameter set to check.

        Returns
        -------
        bool
        """
        if not self.enabled:
            return False

        key = self._make_key(params)
        return self._key_to_path(key).exists()

    def clear(self) -> None:
        """Delete all cached result files.

        Does not remove the cache directory itself.
        """
        if not self.enabled:
            return

        with self._lock:
            cache_path = Path(self.cache_dir)
            deleted = 0
            for pkl_file in cache_path.glob("*.pkl"):
                try:
                    pkl_file.unlink()
                    deleted += 1
                except OSError as exc:
                    logger.warning("Could not delete cache file %s: %s", pkl_file, exc)
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared: %d files deleted", deleted)

    def stats(self) -> Dict[str, int]:
        """Return cache statistics.

        Returns
        -------
        dict
            Keys: ``hits`` (int), ``misses`` (int), ``size`` (int).
            ``size`` is the number of files currently on disk.
        """
        with self._lock:
            size = (
                len(list(Path(self.cache_dir).glob("*.pkl")))
                if self.enabled and Path(self.cache_dir).exists()
                else 0
            )
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": size,
            }

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _make_key(self, params: ParameterSet) -> str:
        """Compute a deterministic MD5 hash for *params*.

        Parameters are sorted by name and rounded to :attr:`hash_precision`
        decimal places before hashing to ensure stability across runs.

        Parameters
        ----------
        params:
            Parameter set to hash.

        Returns
        -------
        str
            32-character lowercase hexadecimal MD5 digest.
        """
        # Round each value and sort by key for determinism
        rounded: Dict[str, float] = {
            k: round(float(v), self.hash_precision)
            for k, v in sorted(params.items())
        }
        canonical = json.dumps(rounded, sort_keys=True)
        return hashlib.md5(canonical.encode("utf-8")).hexdigest()

    def _key_to_path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        return Path(self.cache_dir) / f"{key}.pkl"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SimulationCache(dir={self.cache_dir!r}, "
            f"enabled={self.enabled}, "
            f"hits={s['hits']}, misses={s['misses']}, size={s['size']})"
        )
