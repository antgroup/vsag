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

"""Self-contained HTML visualization for dataset profile reports.

Everything is inlined: CSS, JavaScript, and SVG charts.  Opening the file does
not contact any external service, which keeps the deterministic workflow fully
offline.  Every chart caption states the metric definition, the scope (full
dataset or sample), and the sample size when applicable.
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
)
_MAX_SVG_POINTS = 256


def render_html(report: Dict[str, Any]) -> str:
    """Render a complete standalone HTML document for ``report``."""
    metrics = report.get("metrics", {})
    dataset = report.get("dataset", {})
    config = report.get("config", {})
    file_name = _report_file_name(report)
    body: List[str] = []
    body.append(_header(report, dataset, config))
    body.append(_overview_cards(report, metrics, dataset))
    body.append(_norms_section(metrics.get("norms", {})))
    body.append(_dimensions_section(metrics.get("dimensions", {})))
    body.append(_sparsity_section(metrics.get("sparsity", {})))
    body.append(_duplicates_section(metrics.get("duplicates", {})))
    body.append(_distance_sections(metrics))
    body.append(_anisotropy_section(metrics.get("anisotropy", {})))
    body.append(_clustering_section(metrics.get("clustering", {})))
    body.append(_ai_section(report.get("ai_analysis")))
    body.append(_warnings_section(report.get("warnings", [])))
    body.append(_footer())

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="generator" content="vsag-dataset-profiler">\n'
        "<title>VSAG dataset profile: %s</title>\n"
        "<style>%s</style>\n"
        "</head>\n<body>\n"
        '<div id="tooltip" class="tooltip" role="status" aria-live="polite"></div>\n'
        "%s\n"
        "<script>%s</script>\n"
        "</body>\n</html>\n"
    ) % (
        _e(file_name),
        _CSS,
        "\n".join(part for part in body if part),
        _JS,
    )


def _report_file_name(report: Dict[str, Any]) -> str:
    """Return the input file name stored under ``report.input``."""
    return str(report.get("input", {}).get("file_name") or "dataset")


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def _header(report: Dict[str, Any], dataset: Dict[str, Any], config: Dict[str, Any]) -> str:
    title = "Dataset Feature Profile"
    subtitle = "%s · %s vectors · %s dimensions · %s" % (
        _e(_report_file_name(report)),
        _fmt(dataset.get("count")),
        _fmt(dataset.get("dim")),
        _e(str(dataset.get("vector_type", "dense")).upper()),
    )
    meta = (
        '<span class="pill">schema %s</span>'
        '<span class="pill">seed %s</span>'
        '<span class="pill">sample %s</span>'
        '<span class="pill">pairwise sample %s</span>'
        '<span class="pill">metric %s</span>'
        % (
            _e(str(report.get("schema_version"))),
            _e(str(config.get("seed"))),
            _fmt(config.get("effective_sample_size")),
            _fmt(config.get("effective_pairwise_sample_size")),
            _e(str(config.get("metric"))),
        )
    )
    return (
        '<header class="hero">'
        "<h1>%s</h1><p class=\"subtitle\">%s</p>"
        '<div class="pills">%s</div>'
        '<p class="note">Deterministic offline report. No raw vectors were sent anywhere '
        "while generating this file.</p></header>" % (title, subtitle, meta)
    )


def _overview_cards(
    report: Dict[str, Any], metrics: Dict[str, Any], dataset: Dict[str, Any]
) -> str:
    config = report.get("config", {})
    norms = metrics.get("norms", {}).get("stats", {})
    sparsity = metrics.get("sparsity", {})
    duplicates = metrics.get("duplicates", {}).get("exact", {})
    outliers = metrics.get("outliers", {})
    anisotropy = metrics.get("anisotropy", {})
    invalid = metrics.get("invalid_values", {})
    cards = [
        ("Vectors", _fmt(dataset.get("count")), "full dataset"),
        (
            "Dimensions",
            _fmt(dataset.get("dim")) + (" (inferred)" if dataset.get("dim_inferred") else ""),
            "full dataset",
        ),
        ("Element type", _e(str(dataset.get("dtype"))), "full dataset"),
        (
            "Distance metric",
            _e(str(config.get("metric") or dataset.get("distance_metric") or "n/a")),
            "profiling metric",
        ),
        ("Norm median", _fmt(norms.get("p50")), "full dataset"),
        ("Zero ratio", _pct(sparsity.get("zero_ratio")), "full dataset"),
        (
            "Exact duplicate ratio",
            _pct(duplicates.get("duplicate_ratio")),
            "full dataset",
        ),
        ("Invalid values", _fmt(invalid.get("invalid_values")), "full dataset"),
        ("Norm outlier ratio", _pct(outliers.get("outlier_ratio")), "full dataset"),
        (
            "Top eigenvalue share",
            _fmt(anisotropy.get("top1_eigenvalue_share")),
            "sample",
        ),
    ]
    items = "".join(
        '<div class="card"><div class="card-label">%s</div>'
        '<div class="card-value">%s</div><div class="card-scope">%s</div></div>'
        % (label, value, scope)
        for label, value, scope in cards
    )
    return '<section><h2>Overview</h2><div class="cards">%s</div></section>' % items


def _norms_section(metric: Dict[str, Any]) -> str:
    if not metric:
        return ""
    return _section(
        "Vector norms",
        metric,
        _chart_histogram(metric.get("histogram", {}), "norm", "L2 norm", "Vectors"),
    )


def _dimensions_section(metric: Dict[str, Any]) -> str:
    if not metric:
        return ""
    per_dim = metric.get("per_dimension", {})
    charts = [
        _chart_line(
            per_dim.get("mean", {}),
            "dimension mean",
            "Coordinate",
            "Mean",
            "#2563eb",
        ),
        _chart_line(
            per_dim.get("std", {}),
            "dimension standard deviation",
            "Coordinate",
            "Std",
            "#dc2626",
        ),
        _chart_line(
            per_dim.get("zero_ratio", {}),
            "dimension zero ratio",
            "Coordinate",
            "Zero ratio",
            "#059669",
        ),
    ]
    table = _top_dimension_table(metric)
    return _section(
        "Per-dimension distributions",
        metric,
        "".join(charts) + table,
        extra_note=_downsample_note(per_dim),
    )


def _sparsity_section(metric: Dict[str, Any]) -> str:
    if not metric:
        return ""
    total_cells = (
        metric.get("cells") if "cells" in metric else metric.get("total_values")
    )
    # Sparse zero_values includes implicit zeros; explicit_zero_entries is only
    # a legacy fallback.  Checking key presence also preserves a dense count 0.
    zero_values = (
        metric.get("zero_values")
        if "zero_values" in metric
        else metric.get("explicit_zero_entries")
    )
    rows = [
        ("Representation", str(metric.get("representation"))),
        ("Total cells", _fmt(total_cells)),
        ("Zero values", _fmt(zero_values)),
        ("Non-zero values", _fmt(metric.get("non_zero_values"))),
        ("Density", _pct(metric.get("density"))),
        ("Zero ratio", _pct(metric.get("zero_ratio"))),
        ("All-zero vectors", _fmt(metric.get("all_zero_vectors"))),
    ]
    if metric.get("representation") == "sparse":
        rows.append(("Stored entries", _fmt(metric.get("stored_entries"))))
    charts = ""
    if metric.get("nnz_histogram"):
        charts = _chart_histogram(
            metric["nnz_histogram"], "nnz", "Non-zero entries per vector", "Vectors"
        )
    return _section(
        "Sparsity",
        metric,
        charts,
        extra_note=_table(rows),
    )


def _duplicates_section(metric: Dict[str, Any]) -> str:
    if not metric:
        return ""
    exact = metric.get("exact", {})
    near = metric.get("near_duplicate", {})
    rows = [
        ("Unique vectors", _fmt(exact.get("unique_vectors"))),
        ("Duplicate vectors (exact)", _fmt(exact.get("duplicate_vectors"))),
        ("Duplicate ratio (exact)", _pct(exact.get("duplicate_ratio"))),
        ("Duplicate groups", _fmt(exact.get("duplicate_groups"))),
        ("Largest duplicate group", _fmt(exact.get("max_group_size"))),
    ]
    near_rows = [
        ("Near-duplicate sampled pairs", _fmt(near.get("near_duplicate_pairs"))),
        ("Near-duplicate pair ratio", _pct(near.get("near_duplicate_ratio"))),
        ("Threshold", _fmt(near.get("threshold"))),
        ("Median sampled pairwise distance", _fmt(near.get("median_pairwise_distance"))),
        ("Pairs sampled", _fmt(near.get("pair_count"))),
    ]
    scope_note = metric.get("scope_note", "")
    return _section(
        "Duplicate and near-duplicate indicators",
        metric,
        _table(rows) + "<h3>Near-duplicate (sampled)</h3>" + _table(near_rows),
        extra_note=_e(scope_note),
    )


def _distance_sections(metrics: Dict[str, Any]) -> str:
    pairwise = metrics.get("pairwise_distances", {})
    nearest = metrics.get("nearest_neighbor_distances", {})
    parts: List[str] = []
    if pairwise:
        parts.append(
            _section(
                "Pairwise distance distribution",
                pairwise,
                _chart_histogram(
                    pairwise.get("histogram", {}),
                    "pairwise distance",
                    "Distance between sampled pairs",
                    "Pairs",
                )
                + _metric_stats_table(pairwise.get("stats", {})),
                extra_note="The distance matrix is bounded by the configured pairwise sample size.",
            )
        )
    if nearest and nearest.get("available"):
        parts.append(
            _section(
                "Nearest-neighbor distance distribution",
                nearest,
                _chart_histogram(
                    nearest.get("histogram", {}),
                    "nn distance",
                    "Distance to nearest other sampled vector",
                    "Vectors",
                )
                + _metric_stats_table(nearest.get("stats", {})),
            )
        )
    return "".join(parts)


def _anisotropy_section(metric: Dict[str, Any]) -> str:
    if not metric:
        return ""
    if not metric.get("available"):
        return _section(
            "Anisotropy", metric, '<p class="muted">%s</p>' % _e(metric.get("reason", "unavailable"))
        )
    ratios = metric.get("explained_variance_ratio", []) or []
    charts = _chart_bars(
        ["PC%d" % (index + 1) for index in range(len(ratios))],
        ratios,
        "PC variance share",
        "Explained variance ratio of leading principal components",
        "#7c3aed",
    )
    rows = [
        ("Top-1 eigenvalue share", _fmt(metric.get("top1_eigenvalue_share"))),
        ("Top-10 eigenvalue share", _fmt(metric.get("top10_eigenvalue_share"))),
        ("Effective rank", _fmt(metric.get("effective_rank"))),
        ("Mean pairwise cosine", _fmt(metric.get("mean_pairwise_cosine"))),
    ]
    return _section(
        "Anisotropy",
        metric,
        charts,
        extra_note=_table(rows) + '<p class="note">%s</p>' % _e(metric.get("interpretation", "")),
    )


def _clustering_section(metric: Dict[str, Any]) -> str:
    if not metric:
        return ""
    if not metric.get("available"):
        return _section(
            "Projection and clustering",
            metric,
            '<p class="muted">%s</p>' % _e(metric.get("reason", "unavailable")),
        )
    points = metric.get("projection_points", []) or []
    labels = metric.get("labels") if metric.get("clustering_available") else None
    charts = _chart_scatter(points, labels, "PCA-2 projection of the seeded sample")
    if metric.get("clustering_available"):
        sizes = metric.get("cluster_sizes", []) or []
        charts += _chart_bars(
            ["C%d" % index for index in range(len(sizes))],
            sizes,
            "Cluster size",
            "k-means cluster sizes on the PCA-2 projection",
            "#0891b2",
        )
        charts += _table(
            [
                ("Clusters", _fmt(metric.get("n_clusters"))),
                ("Inertia", _fmt(metric.get("inertia"))),
                ("Iterations", _fmt(metric.get("iterations"))),
            ]
        )
    else:
        charts += '<p class="muted">%s</p>' % _e(metric.get("reason", "clustering disabled"))
    return _section("Projection and clustering", metric, charts)


def _ai_section(analysis: Optional[Dict[str, Any]]) -> str:
    if not analysis:
        return (
            '<section><h2>AI-assisted interpretation</h2>'
            '<p class="muted">Disabled. The deterministic JSON and HTML reports were '
            "generated without network access.</p></section>"
        )
    status = analysis.get("status")
    if status != "ok":
        return (
            '<section><h2>AI-assisted interpretation</h2>'
            '<div class="warning"><strong>AI analysis unavailable.</strong> '
            "The deterministic reports are still complete. %s</div>"
            '<p class="note">Reason: %s</p></section>'
            % (
                _e(str(analysis.get("provider") or "provider")),
                _e(str(analysis.get("error") or "unknown error")),
            )
        )
    findings = analysis.get("findings", []) or []
    recommendations = analysis.get("recommendations", []) or []
    parts = [
        '<section><h2>AI-assisted interpretation</h2>',
        '<p class="note">Provider: %s · model: %s. AI output is derived only from the '
        "aggregate metrics in this report. Recommendations are hypotheses to validate "
        "with reproducible VSAG benchmarks, not guaranteed optimal configurations.</p>"
        % (_e(str(analysis.get("provider"))), _e(str(analysis.get("model")))),
    ]
    if analysis.get("summary"):
        parts.append('<p class="summary">%s</p>' % _e(str(analysis["summary"])))
    parts.append("<h3>Findings</h3>")
    if findings:
        for item in findings:
            parts.append(_ai_item(item, "Finding"))
    else:
        parts.append('<p class="muted">No findings returned.</p>')
    parts.append("<h3>Recommendations to validate</h3>")
    if recommendations:
        for item in recommendations:
            parts.append(_ai_item(item, "Hypothesis"))
    else:
        parts.append('<p class="muted">No recommendations returned.</p>')
    for key, title in (
        ("limitations", "Limitations"),
        ("additional_measurements", "Additional measurements needed"),
    ):
        values = analysis.get(key, []) or []
        if values:
            parts.append("<h3>%s</h3><ul>%s</ul>" % (title, "".join("<li>%s</li>" % _e(str(value)) for value in values)))
    parts.append("</section>")
    return "".join(parts)


def _ai_item(item: Dict[str, Any], label: str) -> str:
    evidence = item.get("evidence", []) or []
    chips = "".join('<code class="evidence">%s</code>' % _e(str(key)) for key in evidence)
    validation = item.get("validation")
    extra = ""
    if validation:
        extra = '<p class="validation"><strong>Validation:</strong> %s</p>' % _e(str(validation))
    confidence = item.get("confidence")
    if confidence:
        extra += '<span class="pill">confidence: %s</span>' % _e(str(confidence))
    return (
        '<article class="ai-item"><h4><span class="ai-label">%s</span> %s</h4>'
        "<p>%s</p><div class=\"evidence-row\">%s</div>%s</article>"
        % (label, _e(str(item.get("title", ""))), _e(str(item.get("detail", ""))), chips, extra)
    )


def _warnings_section(warnings: Sequence[str]) -> str:
    if not warnings:
        return ""
    items = "".join("<li>%s</li>" % _e(str(warning)) for warning in warnings)
    return '<section><h2>Warnings</h2><ul class="warnings">%s</ul></section>' % items


def _footer() -> str:
    return (
        '<footer><p>Generated by <code>vsag-dataset-profiler</code>. Deterministic JSON and '
        "HTML outputs are produced offline. When AI analysis is enabled, only the bounded "
        "aggregate metric payload described in the technical report is sent to the configured "
        "provider; raw vectors and credentials are never included.</p></footer>"
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _section(
    title: str,
    metric: Dict[str, Any],
    chart_html: str,
    extra_note: str = "",
) -> str:
    scope = str(metric.get("scope", "full"))
    sample_size = metric.get("sample_size")
    sample_text = _fmt(sample_size) if sample_size is not None else "n/a"
    definition = metric.get("definition", "")
    note = '<p class="note">%s</p>' % extra_note if extra_note else ""
    return (
        '<section class="metric-section"><h2>%s</h2>'
        '<div class="badges"><span class="badge badge-%s">Scope: %s</span>'
        '<span class="badge">Sample size: %s</span></div>'
        '<p class="definition"><strong>Definition:</strong> %s</p>'
        "%s%s</section>"
        % (_e(title), _e(scope), _e(scope), _e(sample_text), _e(str(definition)), chart_html, note)
    )


def _table(rows: Iterable[Sequence[Any]]) -> str:
    body = "".join(
        "<tr><th>%s</th><td>%s</td></tr>" % (_e(str(row[0])), _e(str(row[1])))
        for row in rows
        if len(row) >= 2
    )
    return '<table class="kv"><tbody>%s</tbody></table>' % body


def _metric_stats_table(stats: Dict[str, Any]) -> str:
    if not stats:
        return ""
    keys = ("count", "mean", "std", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max")
    return _table((key, _fmt(stats.get(key))) for key in keys if key in stats)


def _top_dimension_table(metric: Dict[str, Any]) -> str:
    rows = metric.get("top_variance_dimensions", []) or []
    if not rows:
        return ""
    body = "".join(
        "<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (
            int(row.get("dimension", 0)),
            _fmt(row.get("mean")),
            _fmt(row.get("std")),
            _pct(row.get("zero_ratio")),
        )
        for row in rows
    )
    return (
        "<h3>Highest-variance dimensions</h3>"
        '<table class="grid"><thead><tr><th>Dimension</th><th>Mean</th>'
        "<th>Std</th><th>Zero ratio</th></tr></thead><tbody>%s</tbody></table>" % body
    )


def _downsample_note(per_dim: Dict[str, Any]) -> str:
    reduced = any(
        isinstance(view, dict) and view.get("downsampled")
        for view in per_dim.values()
    )
    if reduced:
        return (
            "Some per-dimension series were bucket-averaged for display; the report "
            "records source_length and bucket_size for each series."
        )
    return ""


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------
def _chart_histogram(
    data: Dict[str, Any], prefix: str, xlabel: str, ylabel: str
) -> str:
    bins = data.get("bins") or []
    counts = data.get("counts") or []
    if len(bins) < 2 or not counts:
        return '<p class="muted">No finite values available for this chart.</p>'
    width, height = 720, 280
    left, right, top, bottom = 56, 16, 18, 48
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(counts) or 1
    bar_width = plot_width / float(len(counts))
    parts = [_svg_open(width, height)]
    parts.append(_axes(left, top, plot_width, plot_height, ylabel))
    for index, count in enumerate(counts):
        bar_height = (count / float(max_count)) * plot_height
        x = left + index * bar_width
        y = top + plot_height - bar_height
        low, high = bins[index], bins[index + 1]
        tip = "%s %.6g – %.6g: %d" % (prefix, low, high, count)
        parts.append(
            '<rect class="bar" x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
            'data-tip="%s"><title>%s</title></rect>'
            % (x, y, max(bar_width - 1.0, 1.0), bar_height, _e(tip), _e(tip))
        )
    parts.append(_x_axis(left, top + plot_height, plot_width, bins[0], bins[-1], xlabel))
    parts.append(_axis_labels(left, top, plot_width, plot_height, max_count))
    parts.append("</svg>")
    return "".join(parts)


def _chart_line(
    data: Dict[str, Any],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> str:
    values = data.get("values") if isinstance(data, dict) else None
    if not values:
        return '<p class="muted">No per-dimension data available.</p>'
    pairs = [(index, float(value)) for index, value in enumerate(values) if value is not None]
    if len(pairs) < 2:
        return '<p class="muted">Not enough per-dimension points for %s.</p>' % _e(title)
    pairs = _thin(pairs, _MAX_SVG_POINTS)
    downsampled = bool(data.get("downsampled"))
    source_length = int(data.get("source_length") or len(values))
    # Match stats.downsample's integer-linspace partition exactly.  A single
    # ceiling bucket size is wrong when source_length is not divisible by the
    # number of output points.
    boundaries = np.linspace(
        0, source_length, len(values) + 1, dtype=np.int64
    )

    def source_range(index: int):
        if not downsampled:
            return index, index
        start = boundaries[index]
        end = boundaries[index + 1] - 1
        return start, max(start, end)

    def x_position(index: int) -> float:
        start, end = source_range(index)
        return (start + end) / 2.0

    width, height = 720, 230
    left, right, top, bottom = 56, 16, 18, 44
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_values = [value for _, value in pairs]
    y_min, y_max = min(y_values), max(y_values)
    if not y_max > y_min:
        # Pad by a representable fraction of the magnitude: adding 1.0 to a
        # value near 1e20 leaves the float64 bound unchanged.
        magnitude = max(abs(y_max), abs(y_min))
        pad = max(magnitude * 1e-3, math.ulp(magnitude) * 4.0, 1e-12)
        y_min -= pad
        y_max += pad
    x_max = max(source_range(index)[1] for index, _ in pairs) or 1

    def sx(index: float) -> float:
        return left + (index / float(x_max)) * plot_width

    def sy(value: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    points = " ".join(
        "%.2f,%.2f" % (sx(x_position(index)), sy(value)) for index, value in pairs
    )
    parts = [_svg_open(width, height)]
    parts.append(_axes(left, top, plot_width, plot_height, ylabel))
    parts.append(
        '<polyline class="line" fill="none" stroke="%s" stroke-width="2" points="%s"/>'
        % (color, points)
    )
    for index, value in pairs:
        start, end = source_range(index)
        if start == end:
            tip = "dimension %d: %.6g" % (start, value)
        else:
            tip = "dimensions %d-%d: %.6g" % (start, end, value)
        parts.append(
            '<circle class="point" cx="%.2f" cy="%.2f" r="2.2" fill="%s" '
            'data-tip="%s"><title>%s</title></circle>'
            % (sx(x_position(index)), sy(value), color, _e(tip), _e(tip))
        )
    parts.append(_x_axis(left, top + plot_height, plot_width, 0, x_max, xlabel))
    parts.append(_axis_labels(left, top, plot_width, plot_height, y_max, y_min))
    parts.append("</svg>")
    caption = (
        '<p class="chart-caption">%s; %d plotted point(s)%s.</p>'
        % (
            _e(title),
            len(pairs),
            "; bucket-averaged" if data.get("downsampled") else "",
        )
    )
    return caption + "".join(parts)


def _chart_bars(
    categories: Sequence[str],
    values: Sequence[Any],
    title: str,
    ylabel: str,
    color: str,
) -> str:
    pairs = [
        (str(category), float(value))
        for category, value in zip(categories, values)
        if value is not None
    ]
    if not pairs:
        return '<p class="muted">No data available for %s.</p>' % _e(title)
    width, height = 720, 230
    left, right, top, bottom = 56, 16, 18, 46
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(value for _, value in pairs)
    if max_value <= 0:
        max_value = 1.0
    bar_width = plot_width / float(len(pairs))
    parts = [_svg_open(width, height), _axes(left, top, plot_width, plot_height, ylabel)]
    for index, (category, value) in enumerate(pairs):
        bar_height = (value / max_value) * plot_height
        x = left + index * bar_width
        y = top + plot_height - bar_height
        tip = "%s: %.6g" % (category, value)
        parts.append(
            '<rect class="bar" x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s" '
            'data-tip="%s"><title>%s</title></rect>'
            % (x, y, max(bar_width - 2.0, 1.0), bar_height, color, _e(tip), _e(tip))
        )
        parts.append(
            '<text class="x-label" x="%.2f" y="%.2f" text-anchor="middle">%s</text>'
            % (x + bar_width / 2.0, top + plot_height + 18, _e(category))
        )
    parts.append(_axis_labels(left, top, plot_width, plot_height, max_value))
    parts.append("</svg>")
    return '<p class="chart-caption">%s.</p>' % _e(title) + "".join(parts)


def _chart_scatter(
    points: Sequence[Sequence[Any]], labels: Optional[Sequence[int]], title: str
) -> str:
    valid = [
        (float(point[0]), float(point[1]), index)
        for index, point in enumerate(points)
        if len(point) >= 2 and point[0] is not None and point[1] is not None
    ]
    if len(valid) < 2:
        return '<p class="muted">Not enough projected points for a scatter plot.</p>'
    width, height = 720, 360
    left, right, top, bottom = 56, 20, 18, 40
    plot_width = width - left - right
    plot_height = height - top - bottom
    xs = [point[0] for point in valid]
    ys = [point[1] for point in valid]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    parts = [_svg_open(width, height)]
    parts.append(_axes(left, top, plot_width, plot_height, "PC2"))
    for x, y, index in valid:
        px = left + ((x - x_min) / (x_max - x_min)) * plot_width
        py = top + plot_height - ((y - y_min) / (y_max - y_min)) * plot_height
        cluster = labels[index] if labels is not None and index < len(labels) else 0
        color = _PALETTE[int(cluster) % len(_PALETTE)]
        tip = "sample %d · cluster %s · (%.4g, %.4g)" % (index, cluster, x, y)
        parts.append(
            '<circle class="point" cx="%.2f" cy="%.2f" r="3.2" fill="%s" opacity="0.8" '
            'data-tip="%s"><title>%s</title></circle>'
            % (px, py, color, _e(tip), _e(tip))
        )
    parts.append(
        '<text class="axis-label" x="%.2f" y="%.2f" text-anchor="middle">PC1</text>'
        % (left + plot_width / 2.0, height - 8)
    )
    parts.append("</svg>")
    return '<p class="chart-caption">%s.</p>' % _e(title) + "".join(parts)


def _svg_open(width: int, height: int) -> str:
    return (
        '<svg class="chart" viewBox="0 0 %d %d" role="img" '
        'preserveAspectRatio="xMidYMid meet">' % (width, height)
    )


def _axes(left: int, top: int, width: int, height: int, ylabel: str) -> str:
    return (
        '<rect class="plot-bg" x="%d" y="%d" width="%d" height="%d"/>'
        '<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>'
        '<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>'
        '<text class="axis-label" transform="translate(14,%.2f) rotate(-90)" '
        'text-anchor="middle">%s</text>'
        % (
            left,
            top,
            width,
            height,
            left,
            top,
            left,
            top + height,
            left,
            top + height,
            left + width,
            top + height,
            top + height / 2.0,
            _e(ylabel),
        )
    )


def _x_axis(
    left: int, y: int, width: int, minimum: float, maximum: float, label: str
) -> str:
    return (
        '<text class="x-label" x="%d" y="%d" text-anchor="start">%s</text>'
        '<text class="x-label" x="%d" y="%d" text-anchor="end">%s</text>'
        '<text class="axis-label" x="%.2f" y="%d" text-anchor="middle">%s</text>'
        % (
            left,
            y + 16,
            _fmt(minimum),
            left + width,
            y + 16,
            _fmt(maximum),
            left + width / 2.0,
            y + 36,
            _e(label),
        )
    )


def _axis_labels(
    left: int,
    top: int,
    width: int,
    height: int,
    maximum: float,
    minimum: float = 0.0,
) -> str:
    return (
        '<text class="y-label" x="%d" y="%d" text-anchor="end">%s</text>'
        '<text class="y-label" x="%d" y="%d" text-anchor="end">%s</text>'
        % (
            left - 6,
            top + 4,
            _fmt(maximum),
            left - 6,
            top + height,
            _fmt(minimum),
        )
    )


def _thin(pairs: List[Any], limit: int) -> List[Any]:
    if len(pairs) <= limit:
        return pairs
    step = int(math.ceil(len(pairs) / float(limit)))
    return pairs[::step]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _e(value: str) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _e(str(value))
    if not math.isfinite(number):
        return "n/a"
    magnitude = abs(number)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e6):
        return "%.4e" % number
    return "%.4g" % number


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return "%.3f%%" % (number * 100.0)


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       "Helvetica Neue", Arial, sans-serif; color: #172033; background: #f5f7fb; }
.hero { padding: 32px 40px 24px; background: linear-gradient(135deg, #172554, #1d4ed8);
        color: #fff; }
.hero h1 { margin: 0 0 6px; font-size: 28px; }
.subtitle { margin: 0 0 14px; opacity: .9; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; }
.pill { padding: 4px 10px; border-radius: 999px; background: rgba(255,255,255,.16);
        font-size: 12px; }
.hero .pill { background: rgba(255,255,255,.16); }
.hero .note { color: rgba(255,255,255,.75); margin: 14px 0 0; font-size: 13px; }
section { margin: 24px 40px; padding: 22px 24px; background: #fff; border-radius: 14px;
          box-shadow: 0 1px 2px rgba(15,23,42,.06); }
h2 { margin: 0 0 10px; font-size: 20px; }
h3 { margin: 18px 0 8px; font-size: 15px; }
h4 { margin: 0 0 6px; font-size: 15px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { padding: 12px 14px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; }
.card-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: .04em; }
.card-value { font-size: 20px; font-weight: 650; margin: 4px 0; word-break: break-word; }
.card-scope { font-size: 11px; color: #94a3b8; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px; }
.badge { padding: 3px 9px; border-radius: 999px; background: #eff6ff; color: #1d4ed8;
         font-size: 12px; font-weight: 600; }
.badge-full { background: #ecfdf5; color: #047857; }
.badge-sample { background: #fff7ed; color: #c2410c; }
.badge-mixed { background: #f5f3ff; color: #6d28d9; }
.definition { color: #475569; font-size: 13px; margin: 0 0 12px; }
.note, .muted { color: #64748b; font-size: 13px; }
.chart { width: 100%; max-width: 900px; height: auto; display: block; margin: 6px 0 18px; }
.chart-caption { color: #475569; font-size: 12px; margin: 2px 0 4px; }
.plot-bg { fill: #fbfdff; }
.axis { stroke: #cbd5e1; stroke-width: 1; }
.bar { fill: #2563eb; opacity: .85; cursor: default; }
.bar:hover, .point:hover { opacity: 1; }
.point { cursor: default; }
.line { stroke-linejoin: round; stroke-linecap: round; }
.axis-label, .x-label, .y-label { fill: #64748b; font-size: 11px; }
table { border-collapse: collapse; margin: 8px 0 12px; font-size: 13px; }
table.kv th { text-align: left; color: #475569; font-weight: 500; padding: 4px 18px 4px 0; }
table.kv td { padding: 4px 0; font-variant-numeric: tabular-nums; }
table.grid { width: 100%; max-width: 760px; }
table.grid th, table.grid td { border-bottom: 1px solid #e2e8f0; padding: 6px 10px;
                               text-align: left; }
table.grid th { color: #64748b; font-weight: 600; }
.warning { padding: 12px 14px; border-radius: 10px; background: #fff7ed; color: #9a3412;
           border: 1px solid #fed7aa; margin: 8px 0; }
.warnings { color: #9a3412; font-size: 13px; }
.summary { font-size: 15px; color: #1e293b; }
.ai-item { border-left: 3px solid #2563eb; padding: 4px 0 4px 14px; margin: 14px 0; }
.ai-label { font-size: 11px; color: #1d4ed8; text-transform: uppercase; letter-spacing: .06em; }
.evidence-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0; }
.evidence { background: #f1f5f9; color: #334155; border-radius: 4px; padding: 2px 6px;
            font-size: 12px; }
.validation { font-size: 13px; color: #475569; }
.tooltip { position: fixed; pointer-events: none; background: #0f172a; color: #fff;
           padding: 5px 8px; border-radius: 6px; font-size: 12px; opacity: 0;
           transition: opacity .08s; z-index: 10; max-width: 320px; }
footer { color: #64748b; font-size: 12px; margin: 8px 40px 40px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
@media (max-width: 640px) { section, footer { margin-left: 12px; margin-right: 12px; }
                            .hero { padding: 24px 16px; } }
"""

_JS = """
(function () {
  var tip = document.getElementById('tooltip');
  document.querySelectorAll('[data-tip]').forEach(function (node) {
    node.addEventListener('mouseenter', function (event) {
      tip.textContent = node.getAttribute('data-tip');
      tip.style.opacity = '1';
    });
    node.addEventListener('mousemove', function (event) {
      tip.style.left = (event.clientX + 14) + 'px';
      tip.style.top = (event.clientY + 14) + 'px';
    });
    node.addEventListener('mouseleave', function () { tip.style.opacity = '0'; });
  });
})();
"""
