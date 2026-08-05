#!/usr/bin/env bash

compatibility_index_dir="${COMPATIBILITY_INDEX_DIR:-/tmp}"
build_dir="${BUILD_DIR:-./build-release}"
compatibility_tool="${build_dir}/tools/check_compatibility/check_compatibility"

if [[ ! -x "${compatibility_tool}" ]]; then
    echo "Error: Compatibility tool not found or not executable at ${compatibility_tool}" >&2
    exit 1
fi

old_version_indexes=()
shopt -s nullglob
for index_file in "${compatibility_index_dir}"/v*_*.index; do
    if [[ -f "$index_file" ]]; then
        name=$(basename "$index_file" .index)
        old_version_indexes+=("$name")
    fi
done
shopt -u nullglob

if [[ ${#old_version_indexes[@]} -eq 0 ]]; then
    echo "Error: No compatibility index files (v*_*.index) found in ${compatibility_index_dir}"
    exit 1
fi

all_success=true
checked_count=0

for version in "${old_version_indexes[@]}"; do
    case "${version,,}" in
        *_hnsw|*_fresh_hnsw|*_diskann)
            echo "Skipping intentionally removed index: $version"
            continue
            ;;
    esac

    ((checked_count += 1))
    echo "Checking compatibility for: $version"
    if ! "${compatibility_tool}" "$version"; then
        echo "Error: Compatibility check failed for $version"
        all_success=false
        break
    fi
done

if [[ $checked_count -eq 0 ]]; then
    echo "Error: No maintained-index compatibility artifacts were found" >&2
    exit 1
elif [ "$all_success" = true ]; then
    echo "All compatibility checks passed"
    exit 0
else
    exit 1
fi
