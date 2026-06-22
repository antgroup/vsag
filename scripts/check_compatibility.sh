#!/usr/bin/env bash

compatibility_index_dir="${COMPATIBILITY_INDEX_DIR:-/tmp}"
compatibility_release_repo="${COMPATIBILITY_RELEASE_REPO:-antgroup/vsag}"
compatibility_release_tag="${COMPATIBILITY_RELEASE_TAG:-compatibility-indexes}"

redownload_asset() {
    local asset_name="$1"
    local asset_path="${compatibility_index_dir}/${asset_name}"

    if ! command -v gh >/dev/null 2>&1; then
        return 1
    fi

    rm -f "${asset_path}"
    mkdir -p "${compatibility_index_dir}"
    gh release download "${compatibility_release_tag}" \
        --repo "${compatibility_release_repo}" \
        --dir "${compatibility_index_dir}" \
        --pattern "${asset_name}" >/dev/null
}

ensure_non_empty_asset() {
    local asset_name="$1"
    local asset_path="${compatibility_index_dir}/${asset_name}"

    if [[ -s "${asset_path}" ]]; then
        return 0
    fi

    echo "Warning: ${asset_path} is missing or empty; attempting to re-download it"
    if redownload_asset "${asset_name}" && [[ -s "${asset_path}" ]]; then
        return 0
    fi

    echo "Error: required compatibility asset is missing or empty: ${asset_path}" >&2
    return 1
}

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

if ! ensure_non_empty_asset "random_512d_10K.bin"; then
    exit 1
fi

all_success=true

for version in "${old_version_indexes[@]}"; do
    echo "Checking compatibility for: $version"
    if ! ensure_non_empty_asset "${version}.index" ||
       ! ensure_non_empty_asset "${version}_build.json" ||
       ! ensure_non_empty_asset "${version}_search.json"; then
        all_success=false
        break
    fi
    if ! ./build-release/tools/check_compatibility/check_compatibility "$version"; then
        echo "Error: Compatibility check failed for $version"
        all_success=false
        break
    fi
done

if [ "$all_success" = true ]; then
    echo "All compatibility checks passed"
    exit 0
else
    exit 1
fi
