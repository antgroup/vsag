"""HGraph backend via pyvsag.

This is the ONLY module in the package that imports pyvsag. Keeping the import
in one place is what makes the "not coupled to pyvsag" property testable: with
pyvsag absent the rest of the kit still imports and runs.

Verified against pyvsag 1.1.0.dev159 on this host (AVX-512):

* ``Index("hgraph", params)`` then ``build(vectors=, ids=, num_elements=, dim=)``.
* ``knn_search`` takes ONE query vector and returns ``(ids, dists)`` -- there is
  no batch entry point, so batching happens here.
* With ``metric_type="cosine"`` the returned values are **distances**
  (identical vectors give ~1e-7), ordered ascending. Callers of this kit expect
  a similarity that is higher-is-better, so scores are converted to
  ``1 - distance``, which for unit-norm vectors is the cosine similarity.
  Getting this backwards would silently return the *worst* neighbours, so it is
  converted once, here, and documented on the interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import IndexBackendUnavailable, IndexStateError
from ..base import IndexStats, SearchHit, VectorIndex

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "pyvsag is an optional dependency and is deliberately not installed by\n"
    "  pip install embeddinggemma2-kit.\n"
    "  Provision it separately (Python 3.10), for example:\n"
    "    uv venv --python 3.10 ~/vsag-py310\n"
    "    uv pip install --python ~/vsag-py310/bin/python pyvsag"
)

# Full-precision vectors. Quantization (sq8/rabitq) would change recall, and the
# point of the baseline comparison is embedding quality, not compression.
_DEFAULT_INDEX_PARAMS: dict[str, Any] = {
    "base_quantization_type": "fp32",
    "max_degree": 64,
    "ef_construction": 400,
}
_DEFAULT_EF_SEARCH = 1000


def is_available() -> bool:
    """Whether pyvsag can be imported in this interpreter.

    Deliberately does NOT touch the logger: importing is enough to answer the
    question, and calling into pyvsag's logger here would set up the state that
    makes a later construction call crash (see ``_silence_logger_once``).
    """
    try:
        import pyvsag  # noqa: F401
    except ImportError:
        return False
    return True


def _import_pyvsag():
    try:
        import pyvsag
    except ImportError as exc:
        raise IndexBackendUnavailable("pyvsag_hgraph", INSTALL_HINT) from exc
    _silence_logger_once(pyvsag)
    return pyvsag


# pyvsag.set_logger_off() is NOT idempotent: calling it a second time segfaults
# the process (reproduced with pyvsag 1.1.0.dev159 on this host -- a single call
# succeeds, a repeat crashes with SIGSEGV). Since both is_available() and index
# construction reach this path, the call is guarded by a module-level flag.
_LOGGER_SILENCED = False


def _silence_logger_once(pyvsag) -> None:
    """Suppress pyvsag's startup banner, at most once per process."""
    global _LOGGER_SILENCED
    if _LOGGER_SILENCED:
        return
    _LOGGER_SILENCED = True
    try:
        pyvsag.set_logger_off()
    except Exception:  # pragma: no cover - logger control is best effort
        pass


class PyvsagHGraphIndex(VectorIndex):
    """Approximate nearest-neighbour index backed by vsag's HGraph."""

    name = "pyvsag_hgraph"

    def __init__(
        self,
        dim: int = 768,
        metric: str = "cosine",
        index_params: dict[str, Any] | None = None,
        ef_search: int = _DEFAULT_EF_SEARCH,
    ) -> None:
        super().__init__(dim=dim, metric=metric)
        if metric not in ("cosine", "ip"):
            raise IndexStateError(
                f"pyvsag HGraph backend supports cosine/ip, got {metric!r}"
            )
        self.ef_search = ef_search
        self.index_params = {**_DEFAULT_INDEX_PARAMS, **(index_params or {})}
        self._index = None
        self._next_id = 0

    # ------------------------------------------------------------ internals

    def _metric_type(self) -> str:
        # Inner product over unit-norm vectors is cosine, and vsag's "cosine"
        # expects the same normalized input. Both map cleanly.
        return "cosine" if self.metric == "cosine" else "ip"

    def _params_json(self) -> str:
        return json.dumps(
            {
                "dtype": "float32",
                "metric_type": self._metric_type(),
                "dim": self.dim,
                "index_param": self.index_params,
            }
        )

    @staticmethod
    def _to_similarity(distances: np.ndarray, metric: str) -> np.ndarray:
        """Convert vsag distances into higher-is-better similarities."""
        # For cosine/ip vsag reports distance = 1 - <a, b>.
        return 1.0 - distances

    # ------------------------------------------------------------ lifecycle

    def build(self, vectors, ids: list[str]) -> None:
        matrix = self._validate_matrix(vectors, ids, "build")
        pyvsag = _import_pyvsag()
        self._index = pyvsag.Index("hgraph", self._params_json())
        self._index.build(
            vectors=matrix,
            ids=list(range(matrix.shape[0])),
            num_elements=matrix.shape[0],
            dim=self.dim,
        )
        self._ids = list(ids)
        self._next_id = matrix.shape[0]
        logger.info(
            "built HGraph index: %d vectors, dim=%d, params=%s",
            matrix.shape[0], self.dim, self.index_params,
        )

    def add(self, vectors, ids: list[str]) -> None:
        matrix = self._validate_matrix(vectors, ids, "add")
        if self._index is None:
            self.build(matrix, ids)
            return
        overlap = set(self._ids) & set(ids)
        if overlap:
            raise IndexStateError(f"ids already present in index: {sorted(overlap)[:5]}")

        # vsag ids are positional ints; the caller-facing string ids are kept
        # in self._ids and mapped back on search.
        new_ids = list(range(self._next_id, self._next_id + matrix.shape[0]))
        self._index.add(vectors=matrix, ids=new_ids)
        self._ids.extend(ids)
        self._next_id += matrix.shape[0]

    def search(self, query, topk: int) -> list[list[SearchHit]]:
        matrix = self._validate_query(query, topk)
        if self._index is None:
            raise IndexStateError("index is empty; call build() or add() before search()")

        k = min(topk, len(self._ids))
        search_params = json.dumps({"hgraph": {"ef_search": max(self.ef_search, k)}})

        results: list[list[SearchHit]] = []
        for row in matrix:
            # knn_search accepts a single vector and returns (ids, dists).
            raw_ids, raw_dists = self._index.knn_search(
                vector=np.ascontiguousarray(row, dtype=np.float32),
                k=k,
                parameters=search_params,
            )
            ids_arr = np.asarray(raw_ids).reshape(-1)
            dists = np.asarray(raw_dists, dtype=np.float32).reshape(-1)
            scores = self._to_similarity(dists, self.metric)

            hits = [
                SearchHit(id=self._ids[int(i)], score=float(s), rank=rank)
                for rank, (i, s) in enumerate(zip(ids_arr, scores))
            ]
            results.append(hits)
        return results

    # ---------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> None:
        if self._index is None:
            raise IndexStateError("cannot save an index that was never built")
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        # vsag owns its own on-disk format.
        self._index.save(str(out / "vsag_index"))
        (out / "ids.json").write_text(json.dumps(self._ids, ensure_ascii=False))
        stats = IndexStats(
            backend=self.name,
            dim=self.dim,
            count=len(self._ids),
            metric=self.metric,
            build_params={**self.index_params, "ef_search": self.ef_search},
        )
        (out / "stats.json").write_text(
            json.dumps(stats.as_dict(), indent=2, ensure_ascii=False)
        )

    def load(self, directory: str | Path) -> None:
        src = Path(directory)
        ids_path = src / "ids.json"
        index_path = src / "vsag_index"
        if not ids_path.exists() or not index_path.exists():
            raise IndexStateError(f"not a saved pyvsag index directory: {src}")

        stats_path = src / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            if stats.get("dim") not in (None, self.dim):
                raise IndexStateError(
                    f"saved index has dim {stats['dim']}, this index expects {self.dim}"
                )

        pyvsag = _import_pyvsag()
        self._index = pyvsag.Index("hgraph", self._params_json())
        self._index.load(str(index_path))
        self._ids = list(json.loads(ids_path.read_text()))
        self._next_id = len(self._ids)