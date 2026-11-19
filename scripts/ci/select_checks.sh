#!/usr/bin/env bash
# YAML parsing is an optimization; unavailable dependencies retain path selection.
set -eu
if python3 -m venv "$RUNNER_TEMP/check-selection" &&
    "$RUNNER_TEMP/check-selection/bin/pip" install --disable-pip-version-check \
        --timeout 15 --retries 1 -r scripts/ci/check-selection-requirements.txt; then
    "$RUNNER_TEMP/check-selection/bin/python" scripts/ci/checks_policy.py
else
    echo '::warning::Optional YAML parser setup failed; retaining original SIMD path selection.'
    SIMD_ACTION_REVISION_FILTER=disabled python3 scripts/ci/checks_policy.py
fi
