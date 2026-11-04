#!/usr/bin/env bash

set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
    echo "ERROR: Git workspace trust may only be configured in GitHub Actions." >&2
    exit 1
fi

if [[ -z "${GITHUB_WORKSPACE:-}" || "$GITHUB_WORKSPACE" != /* ]]; then
    echo "ERROR: GITHUB_WORKSPACE must be an absolute path." >&2
    exit 1
fi

# actions/checkout records this path while using a temporary HOME. Container
# steps run with the job HOME instead, so persist the same exact path there.
git config --global --add safe.directory "$GITHUB_WORKSPACE"
