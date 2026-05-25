#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-/root/data/wholenet-sparse-10m-ip.hdf5}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/benchs/results}"
RUN_ID="${RUN_ID:-sindi_wholenet_sparse_10m_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/$RUN_ID}"
INDEX_PATH="${INDEX_PATH:-$OUTPUT_DIR/sindi.index}"
TOPK="${TOPK:-10}"
SEARCH_QUERY_COUNT="${SEARCH_QUERY_COUNT:-10000}"
SAMPLE_MS="${SAMPLE_MS:-50}"
RERANK_TYPE="${RERANK_TYPE:-fp32}"
DMQ_BITS="${DMQ_BITS:-8}"
QUERY_PRUNE_RATIO="${QUERY_PRUNE_RATIO:-0.2}"
TERM_PRUNE_RATIO="${TERM_PRUNE_RATIO:-0}"
N_CANDIDATE="${N_CANDIDATE:-100}"

if [[ -z "${BUILD_PARAMETER:-}" ]]; then
    read -r -d '' BUILD_PARAMETER <<JSON || true
{
  "dtype": "sparse",
  "dim": 301,
  "metric_type": "ip",
  "index_param": {
    "term_id_limit": 300000,
    "window_size": 50000,
    "doc_prune_ratio": 0.2,
    "use_quantization": true,
    "use_reorder": true,
        "rerank_type": "$RERANK_TYPE",
        "dmq_bits": $DMQ_BITS,
    "avg_doc_term_length": 50
  }
}
JSON
fi

if [[ -z "${SEARCH_PARAMETER:-}" ]]; then
        read -r -d '' SEARCH_PARAMETER <<JSON || true
{
  "sindi": {
        "query_prune_ratio": $QUERY_PRUNE_RATIO,
        "term_prune_ratio": $TERM_PRUNE_RATIO,
        "n_candidate": $N_CANDIDATE,
    "use_term_lists_heap_insert": true
  }
}
JSON
fi

if [[ ! -f "$DATASET" ]]; then
    echo "dataset not found: $DATASET" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$BUILD_PARAMETER" >"$OUTPUT_DIR/build_parameter.json"
printf '%s\n' "$SEARCH_PARAMETER" >"$OUTPUT_DIR/search_parameter.json"

echo "[sindi] result dir: $OUTPUT_DIR"
echo "[sindi] building tools with Makefile"
(
    cd "$ROOT_DIR"
    VSAG_ENABLE_TOOLS=ON make release
)

RUNNER="$ROOT_DIR/build-release/tools/eval/sindi_direct_benchmark"
if [[ ! -x "$RUNNER" ]]; then
    echo "runner not found or not executable: $RUNNER" >&2
    exit 1
fi

BUILD_JSON="$OUTPUT_DIR/build_metrics.json"
SEARCH_JSON="$OUTPUT_DIR/search_metrics.json"

echo "[sindi] build index"
"$RUNNER" \
    --mode build \
    --datapath "$DATASET" \
    --index-path "$INDEX_PATH" \
    --build-parameter "$BUILD_PARAMETER" \
    --output-json "$BUILD_JSON" \
    --sample-ms "$SAMPLE_MS" \
    2>&1 | tee "$OUTPUT_DIR/build.log"

echo "[sindi] search index"
"$RUNNER" \
    --mode search \
    --datapath "$DATASET" \
    --index-path "$INDEX_PATH" \
    --build-parameter "$BUILD_PARAMETER" \
    --search-parameter "$SEARCH_PARAMETER" \
    --output-json "$SEARCH_JSON" \
    --topk "$TOPK" \
    --query-count "$SEARCH_QUERY_COUNT" \
    --sample-ms "$SAMPLE_MS" \
    2>&1 | tee "$OUTPUT_DIR/search.log"

python3 - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
build = json.loads((out_dir / "build_metrics.json").read_text())
search = json.loads((out_dir / "search_metrics.json").read_text())


def human_bytes(value):
    if value is None:
        return "N/A"
    size = float(value)
    unit = "B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


memory_labels = {
    "sindi_object": "SINDI object",
    "window_pointer_array": "Window pointer array",
    "posting_datacells": "Posting data cells",
    "rerank_backend": "Rerank backend",
    "label_table": "Label table",
    "extra_infos": "Extra infos",
    "quantization_params": "Quantization params",
    "__total_size__": "Accounted total",
    "__total_with_common__": "Total with common",
}

rerank_detail_labels = {
    "backend_type": "Backend type",
    "total_bits": "Total bits",
    "id_bits": "ID bits",
    "num_vectors": "Vectors",
    "num_terms": "Stored terms",
    "num_codebooks": "Term codebooks",
    "object": "Backend object",
    "encoded_vectors": "Encoded vectors",
    "id_codes": "Packed IDs",
    "value_codes": "Value codes",
    "codebook_term_ids": "Codebook term IDs",
    "codebooks": "Codebooks",
    "codebook_map_estimate": "Codebook map estimate",
    "codebook_lookup": "Codebook lookup",
    "total": "Backend total",
}

rerank_detail_byte_keys = {
    "object",
    "encoded_vectors",
    "id_codes",
    "value_codes",
    "codebook_term_ids",
    "codebooks",
    "codebook_map_estimate",
    "codebook_lookup",
    "total",
}


def memory_detail_lines(build_detail, search_detail):
    if not isinstance(build_detail, dict) and not isinstance(search_detail, dict):
        return []
    keys = list(memory_labels.keys())
    rows = [
        "## Memory Detail",
        "",
        "| Component | Build | Search-loaded |",
        "| --- | --- | --- |",
    ]
    for key in keys:
        build_value = build_detail.get(key) if isinstance(build_detail, dict) else None
        search_value = search_detail.get(key) if isinstance(search_detail, dict) else None
        rows.append(
            f"| {memory_labels[key]} | {human_bytes(build_value)} | "
            f"{human_bytes(search_value)} |"
        )
    return rows + [""]


def rerank_detail_value(key, value):
    if value is None:
        return "N/A"
    if key in rerank_detail_byte_keys:
        return human_bytes(value)
    return str(value)


def rerank_detail_lines(build_detail, search_detail):
    build_rerank = {}
    search_rerank = {}
    if isinstance(build_detail, dict):
        build_rerank = build_detail.get("rerank_backend_detail", {})
    if isinstance(search_detail, dict):
        search_rerank = search_detail.get("rerank_backend_detail", {})
    if not isinstance(build_rerank, dict) and not isinstance(search_rerank, dict):
        return []
    if not build_rerank and not search_rerank:
        return []

    rows = [
        "## Rerank Backend Detail",
        "",
        "| Component | Build | Search-loaded |",
        "| --- | --- | --- |",
    ]
    for key, label in rerank_detail_labels.items():
        build_value = build_rerank.get(key) if isinstance(build_rerank, dict) else None
        search_value = search_rerank.get(key) if isinstance(search_rerank, dict) else None
        rows.append(
            f"| {label} | {rerank_detail_value(key, build_value)} | "
            f"{rerank_detail_value(key, search_value)} |"
        )
    return rows + [""]


lines = [
    "# SINDI Wholenet Sparse 10M Direct Benchmark",
    "",
    "This run uses the direct `sindi_direct_benchmark` runner, not `eval_performance`.",
    "",
    "## Run",
    "",
    f"- Dataset: `{get(build, ['dataset', 'filepath'])}`",
    f"- Result directory: `{out_dir}`",
    f"- Index path: `{build.get('index_path')}`",
    f"- Base count: {get(build, ['dataset', 'base_count'])}",
    f"- Query count in dataset: {get(build, ['dataset', 'query_count'])}",
    f"- Dim: {get(build, ['dataset', 'dim'])}",
    f"- TopK: {search.get('topk')}",
    f"- Searched queries: {search.get('search_query_count')}",
    "",
    "## Build Metrics",
    "",
    "| Metric | Value |",
    "| --- | --- |",
    f"| Build time | {fmt(build.get('build_seconds'))} s |",
    f"| Build TPS | {fmt(build.get('build_tps'))} vec/s |",
    f"| Index memory | {human_bytes(build.get('index_memory_bytes'))} |",
    f"| Build peak RSS | {human_bytes(build.get('peak_rss_bytes'))} |",
    f"| Build peak RSS delta | {human_bytes(build.get('peak_rss_delta_bytes'))} |",
    f"| Index file size | {human_bytes(build.get('index_file_bytes'))} |",
    f"| Failed IDs | {build.get('failed_id_count')} |",
    "",
    "## Search Metrics",
    "",
    "| Metric | Value |",
    "| --- | --- |",
    f"| Load time | {fmt(search.get('load_seconds'))} s |",
    f"| Search wall time | {fmt(search.get('search_wall_seconds'))} s |",
    f"| QPS | {fmt(search.get('qps'))} |",
    f"| Recall avg | {fmt(search.get('recall_avg'))} |",
    f"| Latency avg | {fmt(search.get('latency_avg_ms'))} ms |",
    f"| Latency p50 | {fmt(get(search, ['latency_detail_ms', 'p50']))} ms |",
    f"| Latency p95 | {fmt(get(search, ['latency_detail_ms', 'p95']))} ms |",
    f"| Latency p99 | {fmt(get(search, ['latency_detail_ms', 'p99']))} ms |",
    f"| Search peak RSS | {human_bytes(search.get('peak_rss_bytes'))} |",
    f"| Search peak RSS delta | {human_bytes(search.get('peak_rss_delta_bytes'))} |",
    f"| Loaded index memory | {human_bytes(search.get('index_memory_bytes'))} |",
    "",
]

lines.extend(
    memory_detail_lines(
        build.get("index_memory_detail"),
        search.get("index_memory_detail"),
    )
)

lines.extend(
    rerank_detail_lines(
        build.get("index_memory_detail"),
        search.get("index_memory_detail"),
    )
)

lines.extend([
    "## Parameters",
    "",
    "Build parameter:",
    "",
    "```json",
    json.dumps(build.get("build_parameter"), indent=2, ensure_ascii=False),
    "```",
    "",
    "Search parameter:",
    "",
    "```json",
    json.dumps(search.get("search_parameter"), indent=2, ensure_ascii=False),
    "```",
    "",
    "## Notes",
    "",
    "- RSS delta is measured after the HDF5 dataset is loaded in that process.",
    "- Build metrics include SINDI build plus index serialization to the saved file.",
    "- Search metrics include index load time separately from query latency/QPS.",
])

(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print(out_dir / "summary.md")
PY

echo "[sindi] summary: $OUTPUT_DIR/summary.md"
