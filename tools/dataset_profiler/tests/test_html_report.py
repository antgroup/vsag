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

from dataset_profiler.html_report import render_html
from dataset_profiler.profiler import profile_dataset
from dataset_profiler.report import DEFINITIONS


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
