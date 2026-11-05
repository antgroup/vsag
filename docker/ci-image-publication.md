# CI image publication on 1.0

After the first successful publish, verify the platform lists for
`ci-v1.0-base`, `ci-v1.0-openblas` (amd64 and arm64) and
`ci-v1.0-mkl` (amd64 only), then switch consumers in a separate PR.
Until that switch lands, consumers still use the shared legacy tags.
Avoid Dockerfile changes in this interval.

Registry maintainers own retention of `ci-v1.0-*-run-<id>-<arch>` tags,
including leftovers from failed builds. Keep tags for any run that may still
publish or be retried. When pruning retired runs, remove only tag aliases
where the registry supports it. Do not delete manifests or package versions
by digest: a live multi-platform index may still reference those manifests.
The workflow itself does not need registry deletion privileges.

The publisher reuses three candidate aliases, one per variant
(`ci-v1.0-{base,openblas,mkl}-candidate`). Registry maintainers retain them for
inspection. Each alias points to the latest candidate for its variant, including
a candidate from a failed verification.

The registry updates one consumer tag at a time. The publisher reads each tag
back and compares its manifest to the verified candidate. If publication stops
partway, the three tags can temporarily refer to different builds. For the
latest publication, rerun the failed publish job with its staging tags intact
to finish promoting the same build. If a newer publication has started, use
the newer run to reconcile all three tags.
