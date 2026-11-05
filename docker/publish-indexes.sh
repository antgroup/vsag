#!/usr/bin/env bash
# Verify every candidate before moving any consumer tag.
set -euo pipefail
meta=$(mktemp)
trap 'rm -f "$meta"' EXIT
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
        docker buildx imagetools inspect --raw "$ref" |
            jq -e --argjson expected "$expected" \
                '([.manifests[].platform | .os + "/" + .architecture] | sort) == $expected'
        targets+=("$target")
        verified+=("$ref")
    done
done
for i in "${!targets[@]}"; do
    docker buildx imagetools create -t "${targets[i]}" "${verified[i]}"
done
