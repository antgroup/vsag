#!/bin/bash

# VSAG requires clang-format version 15 EXACTLY
# Higher or lower versions may produce different formatting

REQUIRED_VERSION="15"

# Find clang-format-15 binary: check the suffixed name first, then use
# 'brew --prefix' for dynamic resolution on macOS, and finally fall back
# to stable Homebrew opt symlinks.
CLANG_FORMAT=""
if command -v clang-format-15 &> /dev/null; then
    CLANG_FORMAT="clang-format-15"
elif command -v brew &> /dev/null; then
    BREW_LLVM_PREFIX=$(brew --prefix llvm@15 2>/dev/null || true)
    if [ -n "$BREW_LLVM_PREFIX" ] && [ -x "$BREW_LLVM_PREFIX/bin/clang-format" ]; then
        CLANG_FORMAT="$BREW_LLVM_PREFIX/bin/clang-format"
    fi
fi
if [ -z "$CLANG_FORMAT" ] && [ -x "/opt/homebrew/opt/llvm@15/bin/clang-format" ]; then
    CLANG_FORMAT="/opt/homebrew/opt/llvm@15/bin/clang-format"
elif [ -z "$CLANG_FORMAT" ] && [ -x "/usr/local/opt/llvm@15/bin/clang-format" ]; then
    CLANG_FORMAT="/usr/local/opt/llvm@15/bin/clang-format"
elif [ -z "$CLANG_FORMAT" ] && command -v clang-format &> /dev/null; then
    CLANG_FORMAT="clang-format"
fi

if [ -z "$CLANG_FORMAT" ]; then
    echo "ERROR: clang-format (version 15) is not installed!"
    echo "  Linux:  sudo apt-get install clang-format-15"
    echo "  macOS:  brew install llvm@15"
    exit 1
fi

# Verify we're using the correct version (portable - no grep -P)
ACTUAL_VERSION=$("$CLANG_FORMAT" --version | sed -n 's/.*version \([0-9]*\).*/\1/p' | head -1)
if [ "$ACTUAL_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: clang-format version mismatch!"
    echo "Required: version $REQUIRED_VERSION"
    echo "Found: version $ACTUAL_VERSION"
    exit 1
fi

echo "Using clang-format version $ACTUAL_VERSION (required: $REQUIRED_VERSION)"

# Format code using the resolved clang-format binary.
find include/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i
find src/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i
find python_bindings/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i
find examples/cpp/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i
find mockimpl/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i
find tests/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i
find tools/ -iname "*.h" -o -iname "*.cpp" | xargs "$CLANG_FORMAT" -i

echo "Code formatting completed with $CLANG_FORMAT"
