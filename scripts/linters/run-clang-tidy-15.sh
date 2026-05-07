#!/bin/bash

# VSAG requires clang-tidy version 15 EXACTLY
# Higher or lower versions may produce different diagnostics

set -euo pipefail

REQUIRED_VERSION="15"

# Find clang-tidy-15 binary: check the suffixed name first, then use
# 'brew --prefix' for dynamic resolution on macOS, and finally fall back
# to stable Homebrew opt symlinks.
CLANG_TIDY=""
CLANG_APPLY=""
if command -v clang-tidy-15 &> /dev/null; then
    CLANG_TIDY="clang-tidy-15"
elif command -v brew &> /dev/null; then
    BREW_LLVM_PREFIX=$(brew --prefix llvm@15 2>/dev/null || true)
    if [ -n "$BREW_LLVM_PREFIX" ] && [ -x "$BREW_LLVM_PREFIX/bin/clang-tidy" ]; then
        CLANG_TIDY="$BREW_LLVM_PREFIX/bin/clang-tidy"
    fi
fi
if [ -z "$CLANG_TIDY" ] && [ -x "/opt/homebrew/opt/llvm@15/bin/clang-tidy" ]; then
    CLANG_TIDY="/opt/homebrew/opt/llvm@15/bin/clang-tidy"
elif [ -z "$CLANG_TIDY" ] && [ -x "/usr/local/opt/llvm@15/bin/clang-tidy" ]; then
    CLANG_TIDY="/usr/local/opt/llvm@15/bin/clang-tidy"
elif [ -z "$CLANG_TIDY" ] && command -v clang-tidy &> /dev/null; then
    CLANG_TIDY="clang-tidy"
fi

if [ -z "$CLANG_TIDY" ]; then
    echo "ERROR: clang-tidy (version 15) is not installed!"
    echo "  Linux:  sudo apt-get install clang-tidy-15"
    echo "  macOS:  brew install llvm@15"
    exit 1
fi

# Verify we're using the correct version (portable - no grep -P)
ACTUAL_VERSION=$("$CLANG_TIDY" --version | sed -n 's/.*version \([0-9]*\).*/\1/p' | head -1)
if [ "$ACTUAL_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: clang-tidy version mismatch!"
    echo "Required: version $REQUIRED_VERSION"
    echo "Found: version $ACTUAL_VERSION"
    exit 1
fi

echo "Using clang-tidy version $ACTUAL_VERSION (required: $REQUIRED_VERSION)"

VERIFY_CONFIG_ARGS=()
RUN_CLANG_TIDY_ARGS=()
EXPECT_CONFIG_VALUE=0
EXPECT_RUN_CONFIG_VALUE=0
for arg in "$@"; do
    if [ "$EXPECT_CONFIG_VALUE" -eq 1 ]; then
        VERIFY_CONFIG_ARGS+=("$arg")
        EXPECT_CONFIG_VALUE=0
    fi

    if [ "$EXPECT_RUN_CONFIG_VALUE" -eq 1 ]; then
        RUN_CLANG_TIDY_ARGS+=("$arg")
        EXPECT_RUN_CONFIG_VALUE=0
        continue
    fi

    case "$arg" in
        --config-file)
            # clang-tidy uses --config-file; run-clang-tidy.py uses -config-file
            VERIFY_CONFIG_ARGS+=("$arg")
            EXPECT_CONFIG_VALUE=1
            RUN_CLANG_TIDY_ARGS+=("-config-file")
            EXPECT_RUN_CONFIG_VALUE=1
            ;;
        -config-file)
            # run-clang-tidy.py form; clang-tidy --verify-config uses --config-file
            VERIFY_CONFIG_ARGS+=("--config-file")
            EXPECT_CONFIG_VALUE=1
            RUN_CLANG_TIDY_ARGS+=("$arg")
            EXPECT_RUN_CONFIG_VALUE=1
            ;;
        -config)
            VERIFY_CONFIG_ARGS+=("$arg")
            EXPECT_CONFIG_VALUE=1
            RUN_CLANG_TIDY_ARGS+=("$arg")
            EXPECT_RUN_CONFIG_VALUE=1
            ;;
        --config-file=*)
            VERIFY_CONFIG_ARGS+=("$arg")
            RUN_CLANG_TIDY_ARGS+=("-config-file=${arg#--config-file=}")
            ;;
        -config-file=*)
            VERIFY_CONFIG_ARGS+=("--config-file=${arg#-config-file=}")
            RUN_CLANG_TIDY_ARGS+=("$arg")
            ;;
        -config=*)
            VERIFY_CONFIG_ARGS+=("$arg")
            RUN_CLANG_TIDY_ARGS+=("$arg")
            ;;
        *)
            RUN_CLANG_TIDY_ARGS+=("$arg")
            ;;
    esac
done

if [ "$EXPECT_CONFIG_VALUE" -eq 1 ]; then
    echo "ERROR: Missing value for clang-tidy config argument."
    exit 1
fi

if [ "$EXPECT_RUN_CONFIG_VALUE" -eq 1 ]; then
    echo "ERROR: Missing value for run-clang-tidy config argument."
    exit 1
fi

# Verify the same clang-tidy config that will be used for linting is parseable.
if ! "$CLANG_TIDY" --verify-config "${VERIFY_CONFIG_ARGS[@]}" 2>&1; then
    echo "ERROR: clang-tidy config is invalid. Fix the config before running lint."
    exit 1
fi

if printf '%s\n' "$@" | grep -qx -- '-fix'; then
    # Find clang-apply-replacements-15 binary
    if command -v clang-apply-replacements-15 &> /dev/null; then
        CLANG_APPLY="clang-apply-replacements-15"
    elif command -v brew &> /dev/null; then
        BREW_LLVM_PREFIX=$(brew --prefix llvm@15 2>/dev/null || true)
        if [ -n "$BREW_LLVM_PREFIX" ] && [ -x "$BREW_LLVM_PREFIX/bin/clang-apply-replacements" ]; then
            CLANG_APPLY="$BREW_LLVM_PREFIX/bin/clang-apply-replacements"
        fi
    fi
    if [ -z "$CLANG_APPLY" ] && [ -x "/opt/homebrew/opt/llvm@15/bin/clang-apply-replacements" ]; then
        CLANG_APPLY="/opt/homebrew/opt/llvm@15/bin/clang-apply-replacements"
    elif [ -z "$CLANG_APPLY" ] && [ -x "/usr/local/opt/llvm@15/bin/clang-apply-replacements" ]; then
        CLANG_APPLY="/usr/local/opt/llvm@15/bin/clang-apply-replacements"
    elif [ -z "$CLANG_APPLY" ] && command -v clang-apply-replacements &> /dev/null; then
        CLANG_APPLY="clang-apply-replacements"
    fi

    if [ -z "$CLANG_APPLY" ]; then
        echo "ERROR: clang-apply-replacements (version 15) is not installed!"
        echo "  Linux:  sudo apt-get install clang-tools-15"
        echo "  macOS:  brew install llvm@15"
        exit 1
    fi

    exec ./scripts/linters/run-clang-tidy.py \
        -clang-tidy-binary "$CLANG_TIDY" \
        -clang-apply-replacements-binary "$CLANG_APPLY" \
        "${RUN_CLANG_TIDY_ARGS[@]}"
fi

    exec ./scripts/linters/run-clang-tidy.py -clang-tidy-binary "$CLANG_TIDY" "${RUN_CLANG_TIDY_ARGS[@]}"
