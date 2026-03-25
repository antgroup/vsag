#!/usr/bin/env bash

old_version_indexes=(
    "v0.17.2_hgraph"
    "v0.17.2_hnsw"
    "v0.16.14_hgraph"
    "v0.16.14_hnsw"
    "v0.15.1_hgraph"
    "v0.15.1_hnsw"
    "v0.14.8_hgraph"
    "v0.14.8_hnsw"
    "v0.13.4_hgraph"
    "v0.13.4_hnsw"
    "v0.13.0_hnsw"
    "v0.12.0_hnsw"
    "v0.11.14_hnsw"
    "v0.10.0_hnsw"
)

all_success=true

for version in "${old_version_indexes[@]}"; do
    echo "Checking compatibility for: $version"
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