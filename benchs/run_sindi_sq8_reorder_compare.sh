#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run SINDI SQ8 posting-list + fp32/DMQ8 reorder comparison with eval_performance.

Required:
  DATAPATH=/path/to/sparse.hdf5

Optional environment variables:
  OUT_DIR=/tmp/vsag_sindi_sq8_reorder_compare
  DIM=512
  TOPK=10
  N_CANDIDATE=200
  SEARCH_QUERY_COUNT=100000
  NUM_THREADS_BUILDING=16
  NUM_THREADS_SEARCHING=16
  TERM_ID_LIMIT=3000000
  WINDOW_SIZE=50000
  DOC_PRUNE_RATIO=0.2
  QUERY_PRUNE_RATIO=0.2
  TERM_PRUNE_RATIO=0.0
  REMAP_TERM_IDS=false
  DELETE_INDEX_AFTER_SEARCH=false
  SKIP_BUILD=0
  EVAL_BIN=/custom/path/eval_performance

Notes:
  - "SQ8" here is SINDI's quantized posting-list storage (`use_quantization: true`).
  - Reorder forward-store variants are fp32 and DMQ8 (`rerank_type: "dmq", dmq_bits: 8`).
  - eval_performance runs max(dataset_query_count, SEARCH_QUERY_COUNT) queries.
  - Each reorder variant is run in a separate eval_performance process so peak-memory
    metrics are isolated.

Example:
  DATAPATH=/data/sparse-full-1M.hdf5 DIM=512 TOPK=10 N_CANDIDATE=100 \
    NUM_THREADS_SEARCHING=32 ./benchs/run_sindi_sq8_reorder_compare.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "${DATAPATH:-}" ]]; then
    echo "error: DATAPATH is required" >&2
    usage >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/vsag_sindi_sq8_reorder_compare}"
EVAL_BIN="${EVAL_BIN:-${ROOT_DIR}/build-release/tools/eval/eval_performance}"

DIM="${DIM:-512}"
TOPK="${TOPK:-10}"
N_CANDIDATE="${N_CANDIDATE:-200}"
SEARCH_QUERY_COUNT="${SEARCH_QUERY_COUNT:-100000}"
NUM_THREADS_BUILDING="${NUM_THREADS_BUILDING:-16}"
NUM_THREADS_SEARCHING="${NUM_THREADS_SEARCHING:-16}"
TERM_ID_LIMIT="${TERM_ID_LIMIT:-3000000}"
WINDOW_SIZE="${WINDOW_SIZE:-50000}"
DOC_PRUNE_RATIO="${DOC_PRUNE_RATIO:-0.2}"
QUERY_PRUNE_RATIO="${QUERY_PRUNE_RATIO:-0.2}"
TERM_PRUNE_RATIO="${TERM_PRUNE_RATIO:-0.0}"
REMAP_TERM_IDS="${REMAP_TERM_IDS:-false}"
DELETE_INDEX_AFTER_SEARCH="${DELETE_INDEX_AFTER_SEARCH:-false}"
SKIP_BUILD="${SKIP_BUILD:-0}"

mkdir -p "${OUT_DIR}/indexes"

if [[ ! -x "${EVAL_BIN}" ]]; then
    echo "[bench] eval_performance not found at ${EVAL_BIN}; building release tools..."
    make -C "${ROOT_DIR}" VSAG_ENABLE_TOOLS=ON release
fi

action_type="build,search"
if [[ "${SKIP_BUILD}" == "1" ]]; then
    action_type="search"
fi

common_search_params="{"
common_search_params+="\"sindi\":{"
common_search_params+="\"query_prune_ratio\":${QUERY_PRUNE_RATIO},"
common_search_params+="\"term_prune_ratio\":${TERM_PRUNE_RATIO},"
common_search_params+="\"n_candidate\":${N_CANDIDATE}"
common_search_params+="}"
common_search_params+="}"

fp32_create_params="{"
fp32_create_params+="\"dim\":${DIM},"
fp32_create_params+="\"dtype\":\"sparse\","
fp32_create_params+="\"metric_type\":\"ip\","
fp32_create_params+="\"index_param\":{"
fp32_create_params+="\"use_quantization\":true,"
fp32_create_params+="\"use_reorder\":true,"
fp32_create_params+="\"rerank_type\":\"fp32\","
fp32_create_params+="\"term_id_limit\":${TERM_ID_LIMIT},"
fp32_create_params+="\"window_size\":${WINDOW_SIZE},"
fp32_create_params+="\"doc_prune_ratio\":${DOC_PRUNE_RATIO},"
fp32_create_params+="\"remap_term_ids\":${REMAP_TERM_IDS}"
fp32_create_params+="}"
fp32_create_params+="}"

dmq8_create_params="{"
dmq8_create_params+="\"dim\":${DIM},"
dmq8_create_params+="\"dtype\":\"sparse\","
dmq8_create_params+="\"metric_type\":\"ip\","
dmq8_create_params+="\"index_param\":{"
dmq8_create_params+="\"use_quantization\":true,"
dmq8_create_params+="\"use_reorder\":true,"
dmq8_create_params+="\"rerank_type\":\"dmq\","
dmq8_create_params+="\"dmq_bits\":8,"
dmq8_create_params+="\"term_id_limit\":${TERM_ID_LIMIT},"
dmq8_create_params+="\"window_size\":${WINDOW_SIZE},"
dmq8_create_params+="\"doc_prune_ratio\":${DOC_PRUNE_RATIO},"
dmq8_create_params+="\"remap_term_ids\":${REMAP_TERM_IDS}"
dmq8_create_params+="}"
dmq8_create_params+="}"

run_case() {
    local case_name="$1"
    local create_params="$2"
    local index_path="$3"

    local config_path="${OUT_DIR}/${case_name}.${timestamp}.yaml"
    local json_out="${OUT_DIR}/${case_name}.${timestamp}.json"
    local table_out="${OUT_DIR}/${case_name}.${timestamp}.md"

    cat >"${config_path}" <<EOF
global:
  num_threads_building: ${NUM_THREADS_BUILDING}
  num_threads_searching: ${NUM_THREADS_SEARCHING}
  exporters:
    print-directly:
      format: "table"
      to: "stdout"
    save-json:
      format: "json"
      to: "file://${json_out}"
    save-table:
      format: "table"
      to: "file://${table_out}"

${case_name}:
  datapath: "${DATAPATH}"
  type: "${action_type}"
  index_name: "sindi"
  create_params: '${create_params}'
  search_params: '${common_search_params}'
  index_path: "${index_path}"
  search_mode: "knn"
  topk: ${TOPK}
  search_query_count: ${SEARCH_QUERY_COUNT}
  delete_index_after_search: ${DELETE_INDEX_AFTER_SEARCH}
EOF

    echo "[bench] case:   ${case_name}"
    echo "[bench] config: ${config_path}"
    echo "[bench] json:   ${json_out}"
    echo "[bench] table:  ${table_out}"
    echo "[bench] index:  ${index_path}"
    echo

    "${EVAL_BIN}" "${config_path}"

    echo
}

timestamp="$(date +%Y%m%d-%H%M%S)"
fp32_index_path="${OUT_DIR}/indexes/sindi_sq8_fp32_reorder.index"
dmq8_index_path="${OUT_DIR}/indexes/sindi_sq8_dmq8_reorder.index"

echo "[bench] action: ${action_type}"
echo "[bench] running each variant in a separate eval_performance process"
echo

run_case "sindi_sq8_fp32_reorder" "${fp32_create_params}" "${fp32_index_path}"
run_case "sindi_sq8_dmq8_reorder" "${dmq8_create_params}" "${dmq8_index_path}"

cat <<EOF >"${OUT_DIR}/sindi_sq8_reorder_compare.${timestamp}.summary"
sindi_sq8_fp32_reorder_json=${OUT_DIR}/sindi_sq8_fp32_reorder.${timestamp}.json
sindi_sq8_dmq8_reorder_json=${OUT_DIR}/sindi_sq8_dmq8_reorder.${timestamp}.json
sindi_sq8_fp32_reorder_table=${OUT_DIR}/sindi_sq8_fp32_reorder.${timestamp}.md
sindi_sq8_dmq8_reorder_table=${OUT_DIR}/sindi_sq8_dmq8_reorder.${timestamp}.md
EOF

echo
echo "[bench] done"
echo "[bench] summary: ${OUT_DIR}/sindi_sq8_reorder_compare.${timestamp}.summary"
