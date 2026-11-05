#!/usr/bin/env bash

set -euo pipefail

compatibility_index_dir="${COMPATIBILITY_INDEX_DIR:-/tmp}"
build_dir="${BUILD_DIR:-./build-release}"
compatibility_tool="${build_dir}/tools/check_compatibility/check_compatibility"

if [[ ! -x "${compatibility_tool}" ]]; then
    echo "Error: Compatibility tool not found or not executable at ${compatibility_tool}" >&2
    exit 1
fi

# Validate the complete input set before executing any compatibility checks.
# Release inventory/digest verification belongs to the caller's trusted manifest.
python3 - "$compatibility_index_dir" <<'PY_VALIDATE'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
try:
    if not root.is_dir():
        raise ValueError(f"Compatibility input directory does not exist: {root}")
    # Published index files define the cases, as in the Community workflow.
    # A release may also contain parameter files whose index is absent from that release.
    removed = ("_hnsw", "_fresh_hnsw", "_diskann")
    index_identifiers = {path.name[:-len(".index")] for path in root.glob("v*_*.index")}
    sidecar_identifiers = set()
    for suffix in ("_build.json", "_search.json"):
        sidecar_identifiers.update(path.name[:-len(suffix)]
                                   for path in root.glob(f"v*_*{suffix}"))
    for name in sorted(sidecar_identifiers - index_identifiers):
        if not name.lower().endswith(removed):
            print(f"Warning: Orphan compatibility parameters without a published index: {name}",
                  file=sys.stderr)
    identifiers = sorted(name for name in index_identifiers
                         if not name.lower().endswith(removed))
    if not identifiers:
        raise ValueError("No maintained-index compatibility artifacts were found")
    vectors = root / "random_512d_10K.bin"
    if not vectors.is_file() or vectors.stat().st_size != 10000 * 512 * 4:
        raise ValueError(f"Expected complete 20480000-byte vector fixture: {vectors}")
    for name in identifiers:
        for suffix in (".index", "_build.json", "_search.json"):
            path = root / (name + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty compatibility input: {path}")
            if suffix.endswith(".json"):
                with path.open() as stream:
                    if not isinstance(json.load(stream), dict):
                        raise ValueError(f"Expected JSON object: {path}")
except (OSError, ValueError) as error:
    print(f"Error: {error}", file=sys.stderr)
    sys.exit(1)
PY_VALIDATE

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
    case "$(printf '%s' "$version" | tr '[:upper:]' '[:lower:]')" in
        *_hnsw|*_fresh_hnsw|*_diskann)
            echo "Skipping intentionally removed index: $version"
            continue
            ;;
    esac

    ((checked_count += 1))
    echo "Checking compatibility for: $version"
    if ! "${compatibility_tool}" "$version" "$compatibility_index_dir"; then
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
