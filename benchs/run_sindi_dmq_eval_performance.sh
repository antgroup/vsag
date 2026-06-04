#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-/root/vsag/data/wholenet_10M/wholenet-10m.hdf5}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/benchs/results}"
RUN_ID="${RUN_ID:-sindi_dmq_eval_performance_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/$RUN_ID}"
TOPK="${TOPK:-10}"
SEARCH_QUERY_COUNT="${SEARCH_QUERY_COUNT:-10000}"
NUM_THREADS_BUILDING="${NUM_THREADS_BUILDING:-16}"
NUM_THREADS_SEARCHING="${NUM_THREADS_SEARCHING:-1}"
BUILD_TOOLS="${BUILD_TOOLS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SEARCH_WARMUP_RUNS="${SEARCH_WARMUP_RUNS:-1}"

EVAL_BIN="${EVAL_BIN:-$ROOT_DIR/build-release/tools/eval/eval_performance}"
RESULT_JSON="$OUTPUT_DIR/eval_results.json"
CONFIG_YAML="$OUTPUT_DIR/eval_config.yaml"
REPORT_MD="$OUTPUT_DIR/summary.md"
CASE_CONFIG_DIR="$OUTPUT_DIR/case_configs"
CASE_RESULTS_DIR="$OUTPUT_DIR/case_results"
WARMUP_RESULTS_DIR="$OUTPUT_DIR/warmup_results"

FP32_INDEX_PATH="${FP32_INDEX_PATH:-$OUTPUT_DIR/fp32/sindi.index}"
DMQ_INDEX_PATH="${DMQ_INDEX_PATH:-$OUTPUT_DIR/dmq/sindi.index}"

if [[ ! -f "$DATASET" ]]; then
    echo "dataset not found: $DATASET" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CASE_CONFIG_DIR" "$CASE_RESULTS_DIR" "$WARMUP_RESULTS_DIR"

if [[ "$BUILD_TOOLS" == "1" ]]; then
    echo "[sindi-dmq] building eval_performance"
    (
        cd "$ROOT_DIR"
        VSAG_ENABLE_TOOLS=ON make release
    )
fi

if [[ ! -x "$EVAL_BIN" ]]; then
    echo "eval_performance not found or not executable: $EVAL_BIN" >&2
    echo "set EVAL_BIN=/path/to/eval_performance or run with BUILD_TOOLS=1" >&2
    exit 1
fi

python3 - "$CONFIG_YAML" "$RESULT_JSON" "$CASE_CONFIG_DIR" "$CASE_RESULTS_DIR" \
    "$WARMUP_RESULTS_DIR" "$DATASET" "$FP32_INDEX_PATH" "$DMQ_INDEX_PATH" "$TOPK" \
    "$SEARCH_QUERY_COUNT" "$NUM_THREADS_BUILDING" "$NUM_THREADS_SEARCHING" <<'PY'
import json
import sys
from pathlib import Path

config_yaml = Path(sys.argv[1])
result_json = Path(sys.argv[2])
case_config_dir = Path(sys.argv[3])
case_results_dir = Path(sys.argv[4])
warmup_results_dir = Path(sys.argv[5])
dataset = sys.argv[6]
fp32_index_path = sys.argv[7]
dmq_index_path = sys.argv[8]
topk = int(sys.argv[9])
search_query_count = int(sys.argv[10])
num_threads_building = int(sys.argv[11])
num_threads_searching = int(sys.argv[12])

base_param = {
    "dtype": "sparse",
    "dim": 301,
    "metric_type": "ip",
    "index_param": {
        "term_id_limit": 300000,
        "window_size": 50000,
        "doc_prune_ratio": 0.2,
        "use_quantization": True,
        "use_reorder": True,
        "avg_doc_term_length": 50,
    },
}

fp32_param = json.loads(json.dumps(base_param))
fp32_param["index_param"]["rerank_type"] = "fp32"

dmq_param = json.loads(json.dumps(base_param))
dmq_param["index_param"]["rerank_type"] = "dmq"
dmq_param["index_param"]["dmq_bits"] = 8

search_param = {
    "sindi": {
        "query_prune_ratio": 0.2,
        "term_prune_ratio": 0,
        "n_candidate": 200,
        "use_term_lists_heap_insert": True,
    }
}


def sq(value):
    return "'" + json.dumps(value, separators=(",", ":")).replace("'", "''") + "'"


cases = [
    ("fp32_build", "build", fp32_param, fp32_index_path),
    ("fp32_search", "search", fp32_param, fp32_index_path),
    ("dmq_build", "build", dmq_param, dmq_index_path),
    ("dmq_search", "search", dmq_param, dmq_index_path),
]


def render_yaml(selected_cases, output_json):
    lines = [
        "global:",
        "  num_threads_building: " + str(num_threads_building),
        "  num_threads_searching: " + str(num_threads_searching),
        "  exporters:",
        "    save-json:",
        f'      to: "file://{output_json}"',
        '      format: "json"',
        "    print-table:",
        '      to: "stdout"',
        '      format: "table"',
        "",
    ]
    for name, action, build_param, index_path in selected_cases:
        lines.extend(
            [
                f"{name}:",
                f'  datapath: "{dataset}"',
                f'  type: "{action}"',
                '  index_name: "sindi"',
                f"  create_params: {sq(build_param)}",
                f"  index_path: \"{index_path}\"",
                f"  topk: {topk}",
                '  search_mode: "knn"',
                "  delete_index_after_search: false",
            ]
        )
        if action == "search":
            lines.extend(
                [
                    f"  search_params: {sq(search_param)}",
                    f"  search_query_count: {search_query_count}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


config_yaml.write_text(render_yaml(cases, result_json))

for case in cases:
    name = case[0]
    case_yaml = case_config_dir / f"{name}.yaml"
    case_json = case_results_dir / f"{name}.json"
    case_yaml.write_text(render_yaml([case], case_json))
    if case[1] == "search":
        warmup_json = warmup_results_dir / f"{name}.warmup.json"
        warmup_yaml = case_config_dir / f"{name}.warmup.yaml"
        warmup_yaml.write_text(render_yaml([case], warmup_json))

(case_config_dir / "README.txt").write_text(
    "Each YAML is executed in a fresh eval_performance process to avoid RSS "
    "peak contamination across cases.\n"
)

(config_yaml.parent / "build_parameter_fp32.json").write_text(
    json.dumps(fp32_param, indent=2) + "\n"
)
(config_yaml.parent / "build_parameter_dmq.json").write_text(
    json.dumps(dmq_param, indent=2) + "\n"
)
(config_yaml.parent / "search_parameter.json").write_text(json.dumps(search_param, indent=2) + "\n")
PY

echo "[sindi-dmq] result dir: $OUTPUT_DIR"
echo "[sindi-dmq] config: $CONFIG_YAML"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[sindi-dmq] dry run enabled; skip eval_performance"
    exit 0
fi

: >"$OUTPUT_DIR/eval_performance.log"
for case_name in fp32_build fp32_search dmq_build dmq_search; do
    case_yaml="$CASE_CONFIG_DIR/$case_name.yaml"
    if [[ "$case_name" == *_search ]]; then
        for warmup_id in $(seq 1 "$SEARCH_WARMUP_RUNS"); do
            warmup_yaml="$CASE_CONFIG_DIR/$case_name.warmup.yaml"
            echo "[sindi-dmq] warmup $case_name run $warmup_id/$SEARCH_WARMUP_RUNS" | tee -a \
                "$OUTPUT_DIR/eval_performance.log"
            "$EVAL_BIN" "$warmup_yaml" 2>&1 | tee -a "$OUTPUT_DIR/eval_performance.log"
        done
    fi
    echo "[sindi-dmq] running $case_name in a fresh process" | tee -a \
        "$OUTPUT_DIR/eval_performance.log"
    "$EVAL_BIN" "$case_yaml" 2>&1 | tee -a "$OUTPUT_DIR/eval_performance.log"
done

python3 - "$RESULT_JSON" "$CASE_RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

result_json = Path(sys.argv[1])
case_results_dir = Path(sys.argv[2])
merged = {}
for name in ["fp32_build", "fp32_search", "dmq_build", "dmq_search"]:
    case_json = case_results_dir / f"{name}.json"
    case_result = json.loads(case_json.read_text())
    if name in case_result:
        merged[name] = case_result[name]
    else:
        merged.update(case_result)
result_json.write_text(json.dumps(merged, separators=(",", ":")) + "\n")
PY

python3 - "$RESULT_JSON" "$REPORT_MD" "$DATASET" "$OUTPUT_DIR" \
    "$FP32_INDEX_PATH" "$DMQ_INDEX_PATH" "$TOPK" "$SEARCH_QUERY_COUNT" \
    "$NUM_THREADS_BUILDING" "$NUM_THREADS_SEARCHING" "$SEARCH_WARMUP_RUNS" <<'PY'
import json
import sys
from pathlib import Path

result_json = Path(sys.argv[1])
report_md = Path(sys.argv[2])
dataset = sys.argv[3]
out_dir = Path(sys.argv[4])
fp32_index_path = Path(sys.argv[5])
dmq_index_path = Path(sys.argv[6])
topk = int(sys.argv[7])
search_query_count = int(sys.argv[8])
num_threads_building = int(sys.argv[9])
num_threads_searching = int(sys.argv[10])
search_warmup_runs = int(sys.argv[11])

results = json.loads(result_json.read_text())


def get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_memory_detail(case):
    value = case.get("memory_detail(B)")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def human_bytes(value):
    if value is None:
        return "N/A"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return "N/A"


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def index_file_size(path):
    return path.stat().st_size if path.exists() else None


def percent_delta(before, after):
    if before in (None, 0) or after is None:
        return "N/A"
    return f"{(after - before) * 100.0 / before:.2f}%"


def ratio_reduction(before, after):
    if before in (None, 0) or after is None:
        return "N/A"
    return f"{(before - after) * 100.0 / before:.2f}%"


def detail_value(detail, key):
    value = detail.get(key)
    if isinstance(value, (int, float)):
        return human_bytes(value)
    if value is None:
        return "N/A"
    return str(value)


fp32_build = results.get("fp32_build", {})
fp32_search = results.get("fp32_search", {})
dmq_build = results.get("dmq_build", {})
dmq_search = results.get("dmq_search", {})

fp32_build_detail = parse_memory_detail(fp32_build)
fp32_search_detail = parse_memory_detail(fp32_search)
dmq_build_detail = parse_memory_detail(dmq_build)
dmq_search_detail = parse_memory_detail(dmq_search)

fp32_index_size = index_file_size(fp32_index_path)
dmq_index_size = index_file_size(dmq_index_path)
fp32_rerank_size = fp32_search_detail.get("rerank_backend")
dmq_rerank_size = dmq_search_detail.get("rerank_backend")

summary_rows = [
    (
        "fp32 rerank",
        fmt(fp32_search.get("qps")),
        fp32_search.get("memory_peak(search)", "N/A"),
        human_bytes(fp32_index_size),
        human_bytes(fp32_rerank_size),
        fmt(fp32_search.get("recall_avg")),
        fmt(fp32_search.get("latency_avg(ms)")),
    ),
    (
        "dmq direct8",
        fmt(dmq_search.get("qps")),
        dmq_search.get("memory_peak(search)", "N/A"),
        human_bytes(dmq_index_size),
        human_bytes(dmq_rerank_size),
        fmt(dmq_search.get("recall_avg")),
        fmt(dmq_search.get("latency_avg(ms)")),
    ),
]

memory_keys = [
    ("sindi_object", "SINDI object"),
    ("window_pointer_array", "Window pointer array"),
    ("posting_datacells", "Posting data cells"),
    ("rerank_backend", "Rerank forward data"),
    ("label_table", "Label table"),
    ("extra_infos", "Extra infos"),
    ("quantization_params", "Quantization params"),
    ("__total_size__", "Accounted total"),
    ("__total_with_common__", "Total with common"),
]

dmq_rerank_detail = dmq_search_detail.get("rerank_backend_detail", {})
dmq_detail_keys = [
    ("backend_type", "Backend type"),
    ("total_bits", "Value bits"),
    ("id_bits", "Term ID bits"),
    ("num_vectors", "Vectors"),
    ("num_terms", "Stored terms"),
    ("num_codebooks", "Codebooks"),
    ("encoded_vectors", "Encoded vector table"),
    ("id_codes", "Packed term IDs"),
    ("value_codes", "Value codes"),
    ("codebook_term_ids", "Codebook term IDs"),
    ("codebooks", "Codebook values"),
    ("codebook_map_estimate", "Codebook map estimate"),
    ("codebook_lookup", "Codebook lookup"),
    ("total", "DMQ rerank total"),
]

lines = [
    "# SINDI DMQ eval_performance 实验报告",
    "",
    "## 实验设置",
    "",
    f"- 数据集：`{dataset}`",
    f"- 结果目录：`{out_dir}`",
    f"- 评测工具：`eval_performance`",
    f"- TopK：{topk}",
    f"- 搜索 Query 数：{search_query_count}",
    f"- 搜索 warmup 次数：{search_warmup_runs}",
    f"- 构建线程数：{num_threads_building}",
    f"- 搜索线程数：{num_threads_searching}",
    f"- fp32 索引：`{fp32_index_path}`",
    f"- DMQ 索引：`{dmq_index_path}`",
    "",
    "## 核心结果",
    "",
    "| 配置 | QPS | Search memory peak | Index size | Rerank 正排数据 size | "
    "Recall avg | Latency avg(ms) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for row in summary_rows:
    lines.append("| " + " | ".join(row) + " |")

lines.extend(
    [
        "",
        "## 对比结论",
        "",
        "| 指标 | fp32 rerank | DMQ direct8 | 变化 |",
        "| --- | ---: | ---: | ---: |",
        f"| QPS | {fmt(fp32_search.get('qps'))} | {fmt(dmq_search.get('qps'))} | "
        f"{percent_delta(fp32_search.get('qps'), dmq_search.get('qps'))} |",
        f"| Index size | {human_bytes(fp32_index_size)} | {human_bytes(dmq_index_size)} | "
        f"{ratio_reduction(fp32_index_size, dmq_index_size)} less |",
        f"| Rerank 正排数据 size | {human_bytes(fp32_rerank_size)} | "
        f"{human_bytes(dmq_rerank_size)} | "
        f"{ratio_reduction(fp32_rerank_size, dmq_rerank_size)} less |",
        f"| Recall avg | {fmt(fp32_search.get('recall_avg'))} | "
        f"{fmt(dmq_search.get('recall_avg'))} | "
        f"{percent_delta(fp32_search.get('recall_avg'), dmq_search.get('recall_avg'))} |",
        "",
        "## Memory Detail",
        "",
        "| Component | fp32 search-loaded | DMQ search-loaded |",
        "| --- | ---: | ---: |",
    ]
)

for key, label in memory_keys:
    lines.append(
        f"| {label} | {detail_value(fp32_search_detail, key)} | "
        f"{detail_value(dmq_search_detail, key)} |"
    )

lines.extend(
    [
        "",
        "## DMQ Rerank Detail",
        "",
        "| Component | Value |",
        "| --- | ---: |",
    ]
)

for key, label in dmq_detail_keys:
    value = dmq_rerank_detail.get(key)
    if key in {
        "encoded_vectors",
        "id_codes",
        "value_codes",
        "codebook_term_ids",
        "codebooks",
        "codebook_map_estimate",
        "codebook_lookup",
        "total",
    }:
        value = human_bytes(value)
    elif value is None:
        value = "N/A"
    lines.append(f"| {label} | {value} |")

lines.extend(
    [
        "",
        "## 构建指标",
        "",
        "| 配置 | Build time(s) | TPS | Memory peak(build) |",
        "| --- | ---: | ---: | ---: |",
        f"| fp32 rerank | {fmt(fp32_build.get('duration(s)'))} | {fmt(fp32_build.get('tps'))} | "
        f"{fp32_build.get('memory_peak(build)', 'N/A')} |",
        f"| dmq direct8 | {fmt(dmq_build.get('duration(s)'))} | {fmt(dmq_build.get('tps'))} | "
        f"{dmq_build.get('memory_peak(build)', 'N/A')} |",
        "",
        "## 参数",
        "",
        "fp32 build parameter:",
        "",
        "```json",
        json.dumps(get(fp32_build, ["index_info"], {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "DMQ build parameter:",
        "",
        "```json",
        json.dumps(get(dmq_build, ["index_info"], {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "Search parameter:",
        "",
        "```json",
        get(dmq_search, ["search_param"], "{}"),
        "```",
        "",
        "## 产物",
        "",
        f"- eval_performance 配置：`{out_dir / 'eval_config.yaml'}`",
        f"- 单 case 配置目录：`{out_dir / 'case_configs'}`",
        f"- 原始 JSON：`{result_json}`",
        f"- 原始日志：`{out_dir / 'eval_performance.log'}`",
        "",
        "## 说明",
        "",
        "- `Search memory peak` 来自 `eval_performance` 的 `memory_peak(search)`。",
        "- 搜索 case 会先执行 warmup，报告只记录 warmup 之后的正式测量结果。",
        "- 每个 case 都单独启动一个 `eval_performance` 进程，避免批量运行时 RSS 互相污染。",
        "- `Index size` 由脚本对序列化后的索引文件执行 `stat` 得到。",
        "- `Rerank 正排数据 size` 使用 SINDI `memory_detail(B).rerank_backend`，即 rerank backend 的内存估算。",
    ]
)

report_md.write_text("\n".join(lines) + "\n")
print(report_md)
PY

echo "[sindi-dmq] report: $REPORT_MD"
