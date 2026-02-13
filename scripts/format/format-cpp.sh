#!/bin/bash

# VSAG requires clang-format version 15 EXACTLY
# Higher or lower versions may produce different formatting

REQUIRED_VERSION="15"

# Check if clang-format-15 is available
if ! command -v clang-format-15 &> /dev/null; then
    echo "ERROR: clang-format-15 is not installed!"
    echo "Please install it with: sudo apt-get install clang-format-15"
    exit 1
fi

# Verify we're using the correct version
ACTUAL_VERSION=$(clang-format-15 --version | grep -oP 'version \K[0-9]+' | head -1)
if [ "$ACTUAL_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: clang-format version mismatch!"
    echo "Required: version $REQUIRED_VERSION"
    echo "Found: version $ACTUAL_VERSION"
    exit 1
fi

echo "Using clang-format version $ACTUAL_VERSION (required: $REQUIRED_VERSION)"

# Format code using clang-format-15
find include/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i
find src/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i
find python_bindings/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i
find examples/cpp/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i
find mockimpl/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i
find tests/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i
find tools/ -iname "*.h" -o -iname "*.cpp" | xargs clang-format-15 -i

echo "Code formatting completed with clang-format-15"
