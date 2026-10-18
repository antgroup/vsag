"""Automatic index construction.

Chooses a backend and parameters from the corpus size, and exposes one entry
point regardless of which backend ends up being used.

Backend policy (explicit, because the two backends differ in recall semantics):

* pyvsag is present -> HGraph.
* pyvsag is absent  -> raise :class:`IndexBackendUnavailable`, UNLESS the caller
  passed ``fallback=True``, which selects the exact numpy backend and logs a
  warning. Silently swapping an approximate index for an exact one (or vice
  versa) would change both latency and recall characteristics without the
  caller knowing, so it is never automatic.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..constants import EMBEDDING_DIM
from ..errors import ConfigError, IndexStateError
from .backends import numpy_flat, pyvsag_hgraph
from .base import SearchHit, VectorIndex

logger = logging.getLogger(__name__)

# Below this many vectors, an exact scan is fast enough that an approximate
# index mainly adds build cost and recall risk.
SMALL_CORPUS_THRESHOLD = 10_000


def _hgraph_params_for(n: int | None) -> dict[str, Any]:
    """Pick HGraph build parameters for an expected corpus size.

    ``max_degree`` controls graph connectivity and ``ef_construction`` the
    build-time search breadth; both dominate achievable recall far more than
    ``ef_search`` does.

    Measured on 20k x 768-d unit-norm vectors, recall@10 vs exact search:

        max_degree=32, ef_construction=200, ef_search=1000 -> 0.956
        max_degree=32, ef_construction=200, ef_search=2000 -> 0.992
        max_degree=64, ef_construction=400, ef_search=1000 -> 1.000

    The low ``max_degree`` values used in vsag's own 128-d examples are not
    enough at 768-d, so the graph is built denser here.
    """
    if n is None or n < 1_000_000:
        return {"max_degree": 64, "ef_construction": 400}
    return {"max_degree": 96, "ef_construction": 600}


# Why ef_search defaults are this high:
# measured on 20k x 768-d unit-norm vectors (the shape real embeddings have),
# recall@10 against exact search is strongly ef_search-bound:
#     ef_search=100  -> 0.32
#     ef_search=400  -> 0.75
#     ef_search=1000 -> 0.95
# High-dimensional vectors make the graph index work much harder than in the
# 128-d examples shipped with vsag, so the low defaults used there are not
# appropriate here.
_DEFAULT_EF_SEARCH = 1000


def _default_ef_search(n: int | None) -> int:
    """Size-aware ef_search default.

    Larger corpora need a wider search to hold the same recall.
    """
    if n is None or n <= 100_000:
        return _DEFAULT_EF_SEARCH
    if n <= 1_000_000:
        return 1500
    return 2000


def create_index(
    dim: int = EMBEDDING_DIM,
    metric: str = "cosine",
    backend: str = "auto",
    fallback: bool = False,
    expected_size: int | None = None,
    ef_search: int | None = None,
    index_params: dict[str, Any] | None = None,
) -> VectorIndex:
    """Create a vector index.

    Args:
        dim: vector dimension; must match the embedder (768).
        metric: ``cosine`` (default) or ``ip``.
        backend: ``auto``, ``hgraph``, or ``numpy``.
        fallback: when True, a missing pyvsag downgrades to the exact numpy
            backend with a warning instead of raising.
        expected_size: planned corpus size, used to tune HGraph parameters.
        ef_search: HGraph search breadth; defaults to a size-aware value.
        index_params: overrides for HGraph build parameters.

    Raises:
        IndexBackendUnavailable: if HGraph was requested (explicitly or via
            ``auto``) but pyvsag is not importable and ``fallback`` is False.
        ConfigError: for an unknown ``backend`` value.
    """
    if backend not in ("auto", "hgraph", "numpy"):
        raise ConfigError(
            f"backend must be 'auto', 'hgraph' or 'numpy', got {backend!r}"
        )

    if backend == "numpy":
        return numpy_flat.NumpyFlatIndex(dim=dim, metric=metric)

    available = pyvsag_hgraph.is_available()

    if backend == "hgraph" and not available and not fallback:
        # Explicit request without fallback: surface the install path.
        from ..errors import IndexBackendUnavailable

        raise IndexBackendUnavailable("pyvsag_hgraph", pyvsag_hgraph.INSTALL_HINT)

    if backend == "auto" and not available and not fallback:
        from ..errors import IndexBackendUnavailable

        raise IndexBackendUnavailable("pyvsag_hgraph", pyvsag_hgraph.INSTALL_HINT)

    if not available:
        logger.warning(
            "pyvsag is unavailable; falling back to the exact numpy backend as "
            "requested (fallback=True). Search will be exact but O(N) per query."
        )
        return numpy_flat.NumpyFlatIndex(dim=dim, metric=metric)

    params = {**_hgraph_params_for(expected_size), **(index_params or {})}
    if ef_search is None:
        # Size-aware default. Do NOT derive this from max_degree: that
        # heuristic yields ~64, which measures 0.32 recall@10 at 768-d.
        ef_search = _default_ef_search(expected_size)
    if expected_size is not None and expected_size < SMALL_CORPUS_THRESHOLD and backend == "auto":
        logger.info(
            "corpus of %d vectors is small; HGraph is still used because pyvsag "
            "is available. Pass backend='numpy' for exact search.",
            expected_size,
        )
    return pyvsag_hgraph.PyvsagHGraphIndex(
        dim=dim, metric=metric, index_params=params, ef_search=ef_search
    )


class AutoIndex:
    """Thin facade that hides backend selection and id bookkeeping.

    Example:
        >>> idx = AutoIndex(expected_size=1000)
        >>> idx.build(vectors, ids)
        >>> hits = idx.search(query_vector, topk=5)

    The backend is resolved on construction so that a missing pyvsag fails
    immediately rather than at first search.
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        metric: str = "cosine",
        backend: str = "auto",
        fallback: bool = False,
        expected_size: int | None = None,
        ef_search: int | None = None,
        index_params: dict[str, Any] | None = None,
    ) -> None:
        self._index = create_index(
            dim=dim,
            metric=metric,
            backend=backend,
            fallback=fallback,
            expected_size=expected_size,
            ef_search=ef_search,
            index_params=index_params,
        )

    @property
    def backend(self) -> str:
        return self._index.name

    @property
    def dim(self) -> int:
        return self._index.dim

    def __len__(self) -> int:
        return len(self._index)

    @property
    def ids(self) -> list[str]:
        return self._index.ids

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        self._index.build(np.asarray(vectors, dtype=np.float32), ids)

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        self._index.add(np.asarray(vectors, dtype=np.float32), ids)

    def search(self, query: np.ndarray, topk: int = 10) -> list[list[SearchHit]]:
        return self._index.search(query, topk)

    def search_one(self, query: np.ndarray, topk: int = 10) -> list[SearchHit]:
        """Convenience for a single query vector."""
        return self.search(query, topk)[0]

    def save(self, directory: str) -> None:
        self._index.save(directory)

    def load(self, directory: str) -> None:
        self._index.load(directory)


def recall_against_exact(
    index: VectorIndex,
    vectors: np.ndarray,
    queries: np.ndarray,
    topk: int = 10,
    ids: list[str] | None = None,
) -> float:
    """Measure ``index`` recall@k against exact brute-force over ``vectors``.

    This is the guard rail for the approximate backend: HGraph is only useful
    if it finds nearly the same neighbours the exact scan does.

    Args:
        index: the index under test. Must have been built over ``vectors`` with
            the same ids.
        vectors: the corpus the index was built from.
        queries: query vectors.
        topk: cut-off for the overlap measurement.
        ids: corpus ids. Defaults to the ids already in ``index``, which is the
            only way the comparison can be correct -- an earlier version of this
            helper generated its own positional ids and silently reported 0.0
            recall whenever the caller used different ids.
    """
    if ids is None:
        ids = index.ids
        if len(ids) != len(vectors):
            raise IndexStateError(
                f"index holds {len(ids)} ids but {len(vectors)} vectors were given; "
                "pass ids explicitly to compare against a freshly built index"
            )

    exact = numpy_flat.NumpyFlatIndex(dim=vectors.shape[1])
    exact.build(vectors, list(ids))

    approx_hits = index.search(queries, topk)
    exact_hits = exact.search(queries, topk)

    overlaps = 0
    total = 0
    for a, e in zip(approx_hits, exact_hits):
        a_ids = {h.id for h in a}
        e_ids = {h.id for h in e}
        overlaps += len(a_ids & e_ids)
        total += len(e_ids)
    return overlaps / total if total else 0.0