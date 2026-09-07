#!/usr/bin/env bash

set -uo pipefail

readonly FAILURE_TAIL_LINES=200

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <log-file> <command> [argument ...]" >&2
    exit 2
fi

log_file="$1"
shift

if ! mkdir -p -- "$(dirname -- "$log_file")"; then
    echo "Unable to create the diagnostic log directory for $log_file" >&2
    exit 2
fi

printf -v command_display '%q ' "$@"
command_display="${command_display% }"

{
    printf 'Command: %s\n' "$command_display"
    echo "Output:"
} | tee "$log_file"
header_statuses=("${PIPESTATUS[@]}")
capture_status="${header_statuses[1]}"

"$@" 2>&1 | tee -a "$log_file"
command_statuses=("${PIPESTATUS[@]}")
command_status="${command_statuses[0]}"
if [[ "${command_statuses[1]}" -ne 0 ]]; then
    capture_status="${command_statuses[1]}"
fi

{
    echo
    printf 'Command: %s\n' "$command_display"
    printf 'Command exit status: %s\n' "$command_status"
    printf 'Diagnostic capture exit status: %s\n' "$capture_status"
} | tee -a "$log_file"
summary_statuses=("${PIPESTATUS[@]}")
if [[ "${summary_statuses[1]}" -ne 0 ]]; then
    capture_status="${summary_statuses[1]}"
fi

if [[ "$command_status" -ne 0 ]]; then
    printf '::error title=Command failed::Exit status %s; full output is in %s\n' \
        "$command_status" "$log_file"
    echo "::group::Failure output (last ${FAILURE_TAIL_LINES} lines)"
    tail -n "$FAILURE_TAIL_LINES" "$log_file" || true
    echo "::endgroup::"
    printf 'Command failed with exit status %s: %s\n' "$command_status" "$command_display"
    exit "$command_status"
fi

if [[ "$capture_status" -ne 0 ]]; then
    printf '::error title=Diagnostic capture failed::tee exited with status %s for %s\n' \
        "$capture_status" "$log_file"
    printf 'Diagnostic capture failed with exit status %s: %s\n' \
        "$capture_status" "$command_display"
    exit "$capture_status"
fi
