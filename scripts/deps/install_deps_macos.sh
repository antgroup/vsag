#!/bin/bash

set -euo pipefail

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Xcode Command Line Tools are required. Run 'xcode-select --install' first."
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required. Install it from https://brew.sh/ first."
    exit 1
fi

formulae=(
    cmake
    ninja
    ccache
    llvm@15
    libomp
    openblas
    gcc
)

for formula in "${formulae[@]}"; do
    if brew list --versions "${formula}" >/dev/null 2>&1; then
        echo "Already installed: ${formula}"
    else
        echo "Installing: ${formula}"
        brew install "${formula}"
    fi
done

echo "macOS build dependencies are installed."
echo "clang-format: $(brew --prefix llvm@15)/bin/clang-format"
echo "clang-tidy: $(brew --prefix llvm@15)/bin/clang-tidy"
echo "libomp: $(brew --prefix libomp)"
echo "openblas: $(brew --prefix openblas)"
