#!/usr/bin/env bash
set -euo pipefail

echo "[install_deps_macos] Starting macOS dependency installation..."

# Ensure Homebrew exists
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Attempt to add brew to PATH for both Intel and Apple Silicon
  if [ -d "/opt/homebrew/bin" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)" || true
  elif [ -d "/usr/local/bin" ]; then
    eval "$(/usr/local/bin/brew shellenv)" || true
  fi
fi

brew update || true

# Packages required for building and running tests on macOS
# Add gcc (for gfortran), openblas (BLAS/LAPACK), lcov (coverage), hdf5, curl
packages=(cmake ninja libomp pkg-config gcc openblas lcov hdf5 curl)
for pkg in "${packages[@]}"; do
  if brew list "$pkg" >/dev/null 2>&1; then
    echo "[install_deps_macos] $pkg already installed"
  else
    echo "[install_deps_macos] Installing $pkg"
    brew install "$pkg" || true
  fi
done

# Optional: ensure cmake and ninja are on PATH (they should be via Homebrew)
command -v cmake >/dev/null 2>&1 || { echo "cmake not found after install"; }
command -v ninja >/dev/null 2>&1 || { echo "ninja not found after install"; }

echo "[install_deps_macos] Done."
