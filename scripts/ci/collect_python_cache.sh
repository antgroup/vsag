#!/bin/bash
# Diagnostics are optional: never turn a successful wheel build into a failure.
# Keep machine-readable counters separate from the portable human-readable fallback.
output=${1:?report directory required}
mkdir -p "$output" || exit 0
ccache --version > "$output/ccache-version.txt" 2>&1 || true
if ! ccache --print-stats > "$output/ccache.tsv" 2> "$output/ccache-stats-error.txt"; then
    rm -f "$output/ccache.tsv"
fi
ccache -s > "$output/ccache.txt" 2>&1 || true
exit 0
