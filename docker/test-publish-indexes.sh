#!/usr/bin/env bash
# Use only an ephemeral loopback registry, never registry credentials.
set -euo pipefail
cd "$(dirname "$0")"
export IMAGE_REPOS="127.0.0.1:${REGISTRY_PORT:?}/test/vsag"
export IMAGE_PREFIX=ci-smoke
export GITHUB_RUN_ID=1
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf 'FROM scratch\nCOPY marker /marker\n' > "$tmp/Dockerfile"
for arch in amd64 arm64; do
    printf '%s\n' "$arch" > "$tmp/marker"
    tag="$IMAGE_REPOS:fixture-$arch"
    docker buildx build --builder default --platform "linux/$arch" --provenance=false --load -t "$tag" "$tmp"
    for variant in base openblas mkl; do
        if [[ "$variant" == mkl && "$arch" == arm64 ]]; then continue; fi
        stage="$IMAGE_REPOS:$IMAGE_PREFIX-$variant-run-$GITHUB_RUN_ID-$arch"
        docker tag "$tag" "$stage"
        docker push "$stage"
    done
done
bash publish-indexes.sh
snapshot() {
    for variant in base openblas mkl; do
        docker buildx imagetools inspect --raw "$IMAGE_REPOS:$IMAGE_PREFIX-$variant" | sha256sum
    done
}
before=$(snapshot)
# Stage a valid change for the first variant and a bad platform for the last.
# Publishing inside the validation loop would incorrectly change the base tag.
printf 'new amd64 base\n' > "$tmp/marker"
stage="$IMAGE_REPOS:$IMAGE_PREFIX-base-run-$GITHUB_RUN_ID-amd64"
docker buildx build --builder default --platform linux/amd64 --provenance=false --load -t "$stage" "$tmp"
docker push "$stage"
stage="$IMAGE_REPOS:$IMAGE_PREFIX-mkl-run-$GITHUB_RUN_ID-amd64"
docker tag "$IMAGE_REPOS:fixture-arm64" "$stage"
docker push "$stage"
if bash publish-indexes.sh; then
    echo 'Invalid MKL platform was accepted' >&2
    exit 1
fi
test "$(snapshot)" = "$before"
