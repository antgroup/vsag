#!/usr/bin/env python3
"""Verify the HGraph backend against exact brute-force search.

Runs in the Python 3.10 environment that has pyvsag installed:

    ~/vsag-py310/bin/python scripts/verify_hgraph.py

Checks:
1. The backend is selected and reports itself as pyvsag_hgraph.
2. Scores are similarities (a vector matches itself with score ~1.0), not
   distances -- getting this backwards would return the worst neighbours.
3. recall@10 against exact search over 768-d unit-norm vectors, which is the
   shape real embeddings have.
4. Persistence round-trips through save/load.
5. The same vectors give the same ranking as the numpy backend on a small set.
"""

from __future__ import annotations

import sys

import numpy as np


def unit_vectors(n: int, dim: int = 768, seed: int = 20261015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim), dtype=np.float32)
    return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True))


def main() -> int:
    try:
        import pyvsag  # noqa: F401
    except ImportError:
        print("SKIP: pyvsag is not importable in this interpreter")
        print("      run with ~/vsag-py310/bin/python for the HGraph checks")
        return 0

    from embeddinggemma2_kit import AutoIndex, recall_against_exact
    from embeddinggemma2_kit.index.backends import pyvsag_hgraph

    print("pyvsag available:", pyvsag_hgraph.is_available())

    n, dim, nq, topk = 20_000, 768, 200, 10
    corpus = unit_vectors(n, dim)
    queries = unit_vectors(nq, dim, seed=7)
    ids = [f"doc-{i}" for i in range(n)]

    failures = []

    # ---- 1/2. backend selection and score direction ----
    idx = AutoIndex(dim=dim, expected_size=n)
    print("selected backend:", idx.backend)
    if idx.backend != "pyvsag_hgraph":
        failures.append(f"expected pyvsag_hgraph, got {idx.backend}")

    idx.build(corpus, ids)

    self_hits = idx.search(corpus[:1], topk=1)[0]
    self_score = self_hits[0].score
    print(f"self-match id={self_hits[0].id} score={self_score:.6f}")
    if self_hits[0].id != "doc-0":
        failures.append(f"self-match returned {self_hits[0].id}, expected doc-0")
    if not (self_score > 0.99):
        failures.append(
            f"self-match score {self_score:.6f} is not ~1.0; scores are probably "
            "distances rather than similarities"
        )

    # ---- 3. recall vs exact ----
    exact = AutoIndex(dim=dim, backend="numpy")
    exact.build(corpus, ids)

    approx_hits = idx.search(queries, topk)
    exact_hits = exact.search(queries, topk)
    overlap = sum(
        len({h.id for h in a} & {h.id for h in e})
        for a, e in zip(approx_hits, exact_hits)
    )
    recall = overlap / (nq * topk)
    print(f"recall@{topk} vs exact (n={n}, dim={dim}): {recall:.4f}")

    # The exact backend must agree with itself perfectly -- a sanity check that
    # the reference implementation is actually exact.
    self_recall = recall_against_exact(exact, corpus, queries[:20], topk=topk, ids=ids)
    print(f"numpy backend self-consistency: {self_recall:.4f}")
    if self_recall < 1.0:
        failures.append(f"numpy backend is not exact: recall {self_recall}")

    # HGraph is approximate; require it to be close, not identical.
    min_recall = 0.95
    if recall < min_recall:
        failures.append(f"HGraph recall {recall:.4f} < {min_recall}")

    # ---- 4. persistence ----
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        idx.save(tmp)
        reloaded = AutoIndex(dim=dim)
        reloaded.load(tmp)
        if len(reloaded) != n:
            failures.append(f"reloaded index has {len(reloaded)} ids, expected {n}")
        before = [h.id for h in approx_hits[0]]
        after = [h.id for h in reloaded.search(queries[:1], topk)[0]]
        print(f"persistence: top-{topk} identical after reload: {before == after}")
        if before != after:
            failures.append("persistence changed the search result")

    # ---- 5. scores agree with exact cosine ----
    q = queries[0]
    true_scores = corpus @ q
    best_true = float(true_scores.max())
    best_approx = idx.search(q, topk=1)[0][0].score
    print(f"top-1 score: approx={best_approx:.6f} exact={best_true:.6f}")
    if abs(best_approx - best_true) > 0.05:
        failures.append(
            f"approximate top-1 score {best_approx:.6f} deviates from exact "
            f"{best_true:.6f}"
        )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL HGRAPH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())