# CI image publication on 1.1

After the first successful publish, verify the platform lists for
`ci-v1.1-base`, `ci-v1.1-openblas` (amd64 and arm64) and
`ci-v1.1-mkl` (amd64 only), then switch consumers in a separate PR.
Until that switch lands, consumers still use the shared legacy tags.
Avoid Dockerfile changes in this interval.

Registry maintainers own retention of `ci-v1.1-*-run-<id>-<arch>` tags,
including leftovers from failed builds. Keep tags for any run that may still
publish or be retried. When pruning retired runs, remove only tag aliases
where the registry supports it. Do not delete manifests or package versions
by digest: a live multi-platform index may still reference those manifests.
The workflow itself does not need registry deletion privileges.
