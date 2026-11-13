# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic index sampling helpers.

Sampling must be reproducible across runs, hosts, and whether or not a sparse
file carries its optional ``*_offsets`` datasets.  All helpers therefore use a
caller-supplied :class:`numpy.random.Generator` seeded with an explicit integer
and do not depend on global NumPy state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SAMPLING_METHOD = (
    "numpy.random.Generator(PCG64); Floyd's algorithm without replacement; "
    "indices sorted ascending before use"
)


def make_rng(seed: int) -> np.random.Generator:
    """Create the deterministic random generator used by the profiler."""
    return np.random.Generator(np.random.PCG64(int(seed)))


def sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``k`` distinct indices from ``[0, n)`` in ascending order.

    Floyd's algorithm is used so that memory and runtime are bounded by ``k``
    rather than by ``n``, which matters for sparse datasets whose offsets are
    not available and which are scanned streaming.  ``k >= n`` returns every
    index.
    """
    n = int(n)
    k = int(k)
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k >= n:
        return np.arange(n, dtype=np.int64)
    selected = set()
    for j in range(n - k, n):
        t = int(rng.integers(0, j + 1))
        if t in selected:
            selected.add(j)
        else:
            selected.add(t)
    return np.array(sorted(selected), dtype=np.int64)


def take_prefix_sample(sample: np.ndarray, limit: Optional[int]) -> np.ndarray:
    """Bound a previously drawn sample by ``limit`` rows.

    ``sample`` is already a uniformly random subset of the dataset, so taking
    its first rows is itself a deterministic uniform sample.  This keeps the
    pairwise-distance phase bounded by ``pairwise_sample_size`` without
    allocating another index set.
    """
    if limit is None:
        return sample
    limit = int(limit)
    if limit <= 0:
        raise ValueError("pairwise sample size must be positive")
    return sample[:limit]
