"""Exact brute-force backend.

This is the reference implementation: it computes every dot product, so its
results are the ground truth that the graph index is measured against. It has
no optional dependencies and is therefore also the thing that keeps this
package importable and usable when pyvsag is absent.

It is intentionally *not* used as a silent substitute for HGraph -- see
:class:`~embeddinggemma2_kit.errors.IndexBackendUnavailable`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from ...errors import IndexStateError
from ..base import IndexStats, SearchHit, VectorIndex


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` without BLAS's spurious floating-point warnings.

    Some BLAS backends (notably Accelerate on macOS arm64) signal ``divide by
    zero`` / ``overflow`` / ``invalid value`` from inside an optimized float32
    matmul even when every output is finite and correct: with unit-norm rows the
    true scores are in [-1, 1], and comparing against a float64 reference gives
    a max difference of ~1e-7 with identical ranking.

    The warning is noise from the kernel's internal blocking, not a property of
    the input. Rather than silencing the category outright -- which would also
    hide a real overflow -- the multiply is computed under ``errstate`` and a
    warning is re-issued if the result is genuinely non-finite.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = a @ b

    if not np.isfinite(result).all():
        warnings.warn(
            "matmul produced non-finite values; inputs may be unnormalized or "
            "contain NaN/inf",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


class NumpyFlatIndex(VectorIndex):
    """Exhaustive inner-product search over a dense matrix."""

    name = "numpy_flat"

    def __init__(self, dim: int = 768, metric: str = "cosine") -> None:
        super().__init__(dim=dim, metric=metric)
        self._matrix: np.ndarray | None = None

    # ------------------------------------------------------------ lifecycle

    def build(self, vectors, ids: list[str]) -> None:
        matrix = self._validate_matrix(vectors, ids, "build")
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._ids = list(ids)

    def add(self, vectors, ids: list[str]) -> None:
        matrix = self._validate_matrix(vectors, ids, "add")
        if self._matrix is None:
            self.build(matrix, ids)
            return
        overlap = set(self._ids) & set(ids)
        if overlap:
            raise IndexStateError(
                f"ids already present in index: {sorted(overlap)[:5]}"
            )
        self._matrix = np.ascontiguousarray(
            np.concatenate([self._matrix, matrix], axis=0), dtype=np.float32
        )
        self._ids.extend(ids)

    def search(self, query, topk: int) -> list[list[SearchHit]]:
        matrix = self._validate_query(query, topk)
        assert self._matrix is not None

        k = min(topk, len(self._ids))
        # scores[q, n] = <query_q, doc_n>; rows are unit norm so this is cosine.
        scores = _matmul(matrix, self._matrix.T)

        # Partial selection then a stable sort of just those k, which keeps
        # ties resolved by ascending index -- matching the reference behavior
        # the baseline evaluation used.
        candidate = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        chosen = np.take_along_axis(scores, candidate, axis=1)
        order = np.argsort(-chosen, axis=1, kind="stable")
        top_scores = np.take_along_axis(chosen, order, axis=1)
        top_indices = np.take_along_axis(candidate, order, axis=1)

        return [
            self._rank_hits(top_scores[row], top_indices[row], self._ids)
            for row in range(matrix.shape[0])
        ]

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        if self._matrix is None:
            raise IndexStateError("cannot save an index that was never built")
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self._matrix)
        (out / "ids.json").write_text(json.dumps(self._ids, ensure_ascii=False))
        stats = IndexStats(
            backend=self.name,
            dim=self.dim,
            count=len(self._ids),
            metric=self.metric,
            build_params={"exact": True},
        )
        (out / "stats.json").write_text(
            json.dumps(stats.as_dict(), indent=2, ensure_ascii=False)
        )

    def load(self, directory: str | Path) -> None:
        src = Path(directory)
        vectors_path = src / "vectors.npy"
        ids_path = src / "ids.json"
        if not vectors_path.exists() or not ids_path.exists():
            raise IndexStateError(f"not a saved index directory: {src}")

        matrix = np.load(vectors_path)
        ids = json.loads(ids_path.read_text())

        stats_path = src / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            if stats.get("dim") not in (None, self.dim):
                raise IndexStateError(
                    f"saved index has dim {stats['dim']}, this index expects {self.dim}"
                )
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._ids = list(ids)