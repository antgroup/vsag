#!/bin/bash

# VSAG requires clang-tidy version 15 EXACTLY
# Higher or lower versions may produce different diagnostics

set -euo pipefail

REQUIRED_VERSION="15"

if ! command -v clang-tidy-15 &> /dev/null; then
    echo "ERROR: clang-tidy-15 is not installed!"
    echo "Please install it with: sudo apt-get install clang-tidy-15"
    exit 1
fi

ACTUAL_VERSION=$(clang-tidy-15 --version | grep -oP 'version \K[0-9]+' | head -1)
if [ "$ACTUAL_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: clang-tidy version mismatch!"
    echo "Required: version $REQUIRED_VERSION"
    echo "Found: version $ACTUAL_VERSION"
    exit 1
fi

echo "Using clang-tidy version $ACTUAL_VERSION (required: $REQUIRED_VERSION)"

if printf '%s\n' "$@" | grep -qx -- '-fix'; then
    if ! command -v clang-apply-replacements-15 &> /dev/null; then
        echo "ERROR: clang-apply-replacements-15 is not installed!"
        echo "Please install it with: sudo apt-get install clang-tools-15"
        exit 1
    fi

    exec ./scripts/linters/run-clang-tidy.py \
        -clang-tidy-binary clang-tidy-15 \
        -clang-apply-replacements-binary clang-apply-replacements-15 \
        "$@"
fi

exec ./scripts/linters/run-clang-tidy.py -clang-tidy-binary clang-tidy-15 "$@"
