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

"""Tests for the self-contained HTML visualization."""

from __future__ import annotations

import re

import numpy as np

from dataset_profiler.html_report import _chart_line, _sparsity_section, render_html
from dataset_profiler.profiler import profile_dataset
from dataset_profiler.report import DEFINITIONS
from dataset_profiler.stats import downsample


def _dense_report(dense_path):
    return profile_dataset(str(dense_path), sample_size=32, pairwise_sample_size=16, clusters=3)


def test_html_is_self_contained(dense_path):
    text = render_html(_dense_report(dense_path))
    assert text.startswith("<!DOCTYPE html>")
    assert "<svg" in text
    assert "vsag-dataset-profiler" in text
    # no external CSS/JS/font resources
    assert "<link" not in text
    assert re.search(r'src="https?://', text) is None
    assert "<script src=" not in text


def test_every_chart_declares_definition_and_scope(dense_path):
    text = render_html(_dense_report(dense_path))
    assert "Definition:" in text
    assert "Scope: full" in text
    assert "Scope: sample" in text
    for name in ("Vector norms", "Per-dimension distributions", "Sparsity"):
        assert name in text
    assert DEFINITIONS["norms"] in text


def test_sparse_html_uses_sparse_sections(sparse_path):
    report = profile_dataset(str(sparse_path), sample_size=16, pairwise_sample_size=8)
    text = render_html(report)
    assert "Non-zero entries per vector" in text
    assert "Representation" in text and "sparse" in text


def test_ai_section_disabled_by_default(dense_path):
    text = render_html(_dense_report(dense_path))
    assert "AI-assisted interpretation" in text
    assert "Disabled" in text


def test_ai_ok_block_is_rendered(dense_path):
    report = _dense_report(dense_path)
    report["ai_analysis"] = {
        "status": "ok",
        "provider": "mock",
        "model": "mock-model",
        "summary": "Highly clustered data with a dominant direction.",
        "findings": [
            {
                "title": "Anisotropic",
                "detail": "The first principal component dominates.",
                "evidence": ["metrics.anisotropy.top1_eigenvalue_share"],
                "confidence": "medium",
            }
        ],
        "recommendations": [
            {
                "title": "Try a graph index",
                "detail": "Hypothesis only.",
                "evidence": ["metrics.pairwise_distances.stats.p50"],
                "validation": "Run tools/eval with HGraph and BruteForce.",
                "confidence": "low",
            }
        ],
        "limitations": ["Sampled metrics only."],
        "additional_measurements": ["Full pairwise distances."],
    }
    text = render_html(report)
    assert "Highly clustered data" in text
    assert "metrics.anisotropy.top1_eigenvalue_share" in text
    assert "Hypothesis" in text
    assert "confidence: low" in text


def test_ai_error_block_is_rendered(dense_path):
    report = _dense_report(dense_path)
    report["ai_analysis"] = {"status": "error", "provider": "mock", "error": "boom"}
    text = render_html(report)
    assert "AI analysis unavailable" in text
    assert "boom" in text


def test_html_escapes_untrusted_strings(dense_path):
    report = _dense_report(dense_path)
    report["ai_analysis"] = {
        "status": "error",
        "provider": "mock",
        "error": "<script>alert(1)</script>",
    }
    text = render_html(report)
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_warnings_are_rendered(sparse_no_offsets_path):
    report = profile_dataset(str(sparse_no_offsets_path), sample_size=8, pairwise_sample_size=8)
    text = render_html(report)
    assert "Warnings" in text


def test_page_title_and_header_use_input_file_name(dense_path):
    text = render_html(_dense_report(dense_path))
    assert "<title>VSAG dataset profile: dense_toy.hdf5</title>" in text
    assert "dense_toy.hdf5 ·" in text


def test_overview_shows_effective_metric(dense_path):
    # the file attribute is 'l2' (euclidean) but the run used angular
    report = profile_dataset(
        str(dense_path), metric="angular", sample_size=8, pairwise_sample_size=8
    )
    text = render_html(report)
    assert '<div class="card-value">angular</div>' in text


def test_large_constant_range_renders_with_representable_padding(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "huge_range.hdf5"
    with h5py.File(str(path), "w") as handle:
        handle.attrs["distance"] = "euclidean"
        handle.create_dataset(
            "train",
            data=np.array([[1e20, 0.0], [0.0, 2e20], [2e20, 1e20]], dtype=np.float32),
        )
    report = profile_dataset(
        str(path), sample_size=3, pairwise_sample_size=3, clusters=2
    )
    # per-dimension means are constant at ~1e20, so the chart range must be
    # padded by a representable amount instead of +/-1.0
    assert report["metrics"]["dimensions"]["per_dimension"]["mean"]["values"][0] == (
        report["metrics"]["dimensions"]["per_dimension"]["mean"]["values"][1]
    )
    text = render_html(report)
    assert "Per-dimension distributions" in text
    assert re.search(r'(cx|cy)="[^"]*(nan|inf)', text) is None


def test_downsampled_points_labelled_with_source_dimension_ranges(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "bucket.hdf5"
    with h5py.File(str(path), "w") as handle:
        handle.attrs["distance"] = "euclidean"
        handle.create_dataset(
            "train",
            data=np.arange(16, dtype=np.float32).reshape(4, 4),
        )
    report = profile_dataset(
        str(path),
        sample_size=4,
        pairwise_sample_size=4,
        max_dimension_points=2,
    )
    assert report["metrics"]["dimensions"]["per_dimension"]["mean"]["downsampled"] is True
    text = render_html(report)
    assert "dimensions 0-1" in text
    assert "dimensions 2-3" in text


def test_downsampled_points_use_exact_nonuniform_source_ranges(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "nonuniform_buckets.hdf5"
    with h5py.File(str(path), "w") as handle:
        handle.attrs["distance"] = "euclidean"
        handle.create_dataset(
            "train", data=np.arange(10, dtype=np.float32).reshape(2, 5)
        )
    report = profile_dataset(
        str(path), sample_size=2, pairwise_sample_size=2, max_dimension_points=2
    )
    text = render_html(report)
    assert "dimensions 0-1" in text
    assert "dimensions 2-4" in text
    assert "dimensions 0-2" not in text
    assert "dimensions 3-4" not in text


def test_downsampled_points_use_the_exact_linspace_partition():
    data = downsample(np.arange(784, dtype=np.float64), max_points=768)
    text = _chart_line(data, "dimension mean", "Coordinate", "Mean", "#2563eb")
    # Bucket 240 is where exact integer division and integer linspace disagree.
    assert "dimensions 244-245" in text
    assert "dimension 245:" not in text


def test_sparse_zero_total_includes_implicit_zeros(sparse_path):
    report = profile_dataset(str(sparse_path), sample_size=8, pairwise_sample_size=4)
    sparsity = report["metrics"]["sparsity"]
    expected = sparsity["cells"] - sparsity["non_zero_values"]
    assert sparsity["zero_values"] == expected
    assert sparsity["explicit_zero_entries"] == 0
    text = render_html(report)
    assert "<th>Zero values</th><td>%d</td>" % expected in text


def test_dense_zero_count_is_not_rendered_as_unavailable():
    text = _sparsity_section(
        {
            "representation": "dense",
            "total_values": 1,
            "zero_values": 0,
            "non_zero_values": 1,
        }
    )
    assert "<th>Zero values</th><td>0</td>" in text

