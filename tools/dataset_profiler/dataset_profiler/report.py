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

"""Versioned JSON report assembly and metric definitions.

Every rendered metric carries a human-readable ``definition``, a ``scope``
(``"full"`` or ``"sample"``), and the ``sample_size`` used when the scope is a
sample.  This keeps full-dataset and sampled statistics visibly distinct in
both the JSON report and the HTML visualization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION, TOOL_NAME, __version__
from .hdf5 import DatasetMetadata

DEFINITIONS: Dict[str, str] = {
    "norms": (
        "L2 norm ||x||_2 of every vector, computed over all finite values in the "
        "selected split."
    ),
    "dimensions": (
        "Per-coordinate count, mean, standard deviation, minimum, maximum, and "
        "zero ratio over the selected split."
    ),
    "sparsity": (
        "Count and fraction of exactly-zero entries; for sparse vectors this also "
        "reports non-zero entries per vector and density = nnz / (count * dim)."
    ),
    "duplicates": (
        "Exact duplicate detection by hashing each complete vector; near-duplicate "
        "pairs use Euclidean distance for inner-product inputs and the chosen "
        "distance otherwise, below a relative threshold derived from the sampled "
        "duplicate-detection distance median."
    ),
    "pairwise_distances": (
        "Distribution of distances between distinct vectors in a bounded random "
        "sample. The full M x M matrix is allocated only for the configured sample "
        "size, never for the complete dataset."
    ),
    "nearest_neighbor_distances": (
        "For each vector in the bounded sample, the distance to its nearest other "
        "vector in the same sample (diagonal excluded)."
    ),
    "invalid_values": (
        "Counts of NaN/Inf values in the selected split and the number of vectors "
        "containing at least one such value. Invalid entries are excluded from "
        "numeric summaries and treated as zero in norms."
    ),
    "outliers": (
        "Robust norm outliers: vectors whose norm has a modified z-score above 3.5, "
        "using median and median absolute deviation. Invalid (NaN/Inf) value counts "
        "are reported separately."
    ),
    "anisotropy": (
        "Eigenvalue concentration of the centered sample covariance: the largest "
        "eigenvalue share, top-10 share, effective rank, and mean pairwise cosine "
        "similarity. High top-1 share indicates a dominant direction."
    ),
    "clustering": (
        "Optional deterministic PCA-2 projection of the sample and optional k-means "
        "cluster summary over that projection."
    ),
}


def metric_entry(
    definition: str,
    scope: str,
    sample_size: Optional[int] = None,
    **fields: Any
) -> Dict[str, Any]:
    """Create a metric dictionary with its provenance metadata."""
    entry: Dict[str, Any] = {
        "definition": definition,
        "scope": scope,
        "sample_size": sample_size,
    }
    entry.update(fields)
    return entry


def build_report(
    metadata: DatasetMetadata,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    """Assemble the deterministic, versioned profile report."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "name": TOOL_NAME,
            "version": __version__,
        },
        "input": {
            "file_name": metadata.file_name,
            "file_size_bytes": metadata.file_size_bytes,
            "split": metadata.split,
        },
        "dataset": {
            "vector_type": metadata.vector_type,
            "dtype": metadata.dtype,
            "count": metadata.count,
            "dim": metadata.dim,
            "dim_inferred": metadata.dim_inferred,
            "distance_metric": metadata.distance,
            "has_offsets": metadata.has_offsets,
            "datasets": list(metadata.datasets),
            "attributes": dict(metadata.attributes),
        },
        "config": dict(config),
        "metrics": metrics,
        "warnings": list(warnings),
    }
