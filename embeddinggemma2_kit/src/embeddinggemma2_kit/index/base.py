"""Backend-agnostic vector index interface.

The kit ships two backends with deliberately different guarantees:

* ``pyvsag_hgraph`` -- approximate graph index (HGraph) via pyvsag. Fast, and the
  intended production path. Requires the optional pyvsag dependency.
* ``numpy_flat`` -- exact brute-force dot product. Used as the reference for
  recall measurement and as an explicit opt-in fallback.

Because those two have different recall semantics, the kit never swaps one for
the other silently. See :class:`~embeddinggemma2_kit.errors.IndexBackendUnavailable`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import EMBEDDING_DIM
from ..errors import IndexStateError


@dataclass
class SearchHit:
    """A single retrieved neighbour."""

    id: str
    score: float
    rank: int


@dataclass
class IndexStats:
    """Bookkeeping written alongside a persisted index for reproducibility."""

    backend: str
    dim: int
    count: int = 0
    metric: str = "cosine"
    build_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "count": self.count,
            "metric": self.metric,
            "build_params": self.build_params,
            "metadata": self.metadata,
        }


class VectorIndex(ABC):
    """Common interface implemented by every backend.

    Implementations accept L2-normalised float32 vectors, so an inner-product
    search is equivalent to cosine similarity.
    """

    name: str = "base"

    def __init__(self, dim: int = EMBEDDING_DIM, metric: str = "cosine") -> None:
        if dim <= 0:
            raise IndexStateError(f"dim must be positive, got {dim}")
        if metric not in ("cosine", "ip", "l2"):
            raise IndexStateError(f"unsupported metric {metric!r}")
        self.dim = dim
        self.metric = metric
        self._ids: list[str] = []

    # ------------------------------------------------------------- lifecycle

    @abstractmethod
    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Create the index from scratch, replacing any existing contents."""

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Append vectors to an existing index."""

    @abstractmethod
    def search(self, query: np.ndarray, topk: int) -> list[list[SearchHit]]:
        """Search one or more queries. Returns one hit list per query row."""

    @abstractmethod
    def save(self, directory: str) -> None:  # pragma: no cover - backend specific
        ...

    @abstractmethod
    def load(self, directory: str) -> None:  # pragma: no cover - backend specific
        ...

    # ---------------------------------------------------------------- shared

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def _validate_matrix(self, vectors: Any, ids: list[str] | None, what: str) -> np.ndarray:
        """Shared shape/dtype/finiteness checks for build and add."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise IndexStateError(
                f"{what} expects a 2-D array, got shape {matrix.shape}"
            )
        if matrix.shape[1] != self.dim:
            raise IndexStateError(
                f"{what} expects {self.dim}-d vectors, got {matrix.shape[1]}"
            )
        if matrix.shape[0] == 0:
            raise IndexStateError(f"{what} received an empty batch")
        if not np.isfinite(matrix).all():
            raise IndexStateError(f"{what} received non-finite values")
        if ids is not None:
            if len(ids) != matrix.shape[0]:
                raise IndexStateError(
                    f"{what} got {len(ids)} ids for {matrix.shape[0]} vectors"
                )
            duplicates = {i for i in ids if ids.count(i) > 1} if len(ids) < 10_000 else set()
            if duplicates:
                raise IndexStateError(f"{what} received duplicate ids: {sorted(duplicates)[:5]}")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _validate_query(self, query: Any, topk: int) -> np.ndarray:
        if topk <= 0:
            raise IndexStateError(f"topk must be positive, got {topk}")
        if len(self._ids) == 0:
            raise IndexStateError("index is empty; call build() or add() before search()")
        matrix = np.asarray(query, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise IndexStateError(
                f"query must be ({self.dim},) or (N, {self.dim}), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise IndexStateError("query contains non-finite values")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    @staticmethod
    def _rank_hits(scores: np.ndarray, indices: np.ndarray, id_list: list[str]) -> list[SearchHit]:
        return [
            SearchHit(id=id_list[int(i)], score=float(s), rank=rank)
            for rank, (s, i) in enumerate(zip(scores, indices))
        ]