#!/usr/bin/env bash
# Verify every candidate before moving any consumer tag.
set -euo pipefail
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
meta="$work/metadata.json"
targets=()
verified=()
for repo in $IMAGE_REPOS; do
    for variant in base openblas mkl; do
        target="$repo:$IMAGE_PREFIX-$variant"
        sources=("$target-run-$GITHUB_RUN_ID-amd64")
        expected='["linux/amd64"]'
        if [[ "$variant" != mkl ]]; then
            sources+=("$target-run-$GITHUB_RUN_ID-arm64")
            expected='["linux/amd64","linux/arm64"]'
        fi
        docker buildx imagetools create --prefer-index=true --metadata-file "$meta" \
            -t "$target-candidate" "${sources[@]}"
        digest=$(jq -er '."containerimage.descriptor".digest | select(test("^sha256:[0-9a-f]{64}$"))' "$meta")
        ref="$repo@$digest"
        manifest="$work/verified-${#targets[@]}.json"
        docker buildx imagetools inspect --raw "$ref" > "$manifest"
        jq -e --argjson expected "$expected" \
            '([.manifests[].platform | .os + "/" + .architecture] | sort) == $expected' "$manifest"
        targets+=("$target")
        verified+=("$ref")
    done
done
for i in "${!targets[@]}"; do
    target=${targets[i]}
    docker buildx imagetools create -t "$target" "${verified[i]}"
    docker buildx imagetools inspect --raw "$target" > "$work/published.json"
    if ! cmp -s "$work/verified-$i.json" "$work/published.json"; then
        echo "Published index for $target differs from ${verified[i]}" >&2
        exit 1
    fi
done
