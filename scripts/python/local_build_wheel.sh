#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR=".venv"
SUPPORTED_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12")
HAVE_DOCKER=true

# --- Prerequisite Checks ---
echo "🔎 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 'python3' command not found. Please ensure Python 3 is installed."
    exit 1
fi
if [ ! -f "scripts/python/prepare_python_build.sh" ]; then
    echo "❌ Preparation script not found at 'scripts/python/prepare_python_build.sh'."
    exit 1
fi
if ! docker info > /dev/null 2>&1; then
  echo "⚠️ Docker daemon is not running. Build without Docker."
  HAVE_DOCKER=false
fi
echo "✅ Prerequisites met."

# --- Virtual Environment Setup ---
echo "🔎 Setting up Python virtual environment..."
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
  # Not in any virtual environment, create and activate .venv
  if [ ! -d "$VENV_DIR" ]; then
    echo "   - Virtual environment not found, creating at './${VENV_DIR}'..."
    python3 -m venv "$VENV_DIR"
  fi
  source "${VENV_DIR}/bin/activate"
  echo "✅ Virtual environment activated."
else
  if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
  elif [ -n "$CONDA_PREFIX" ]; then
    echo "✅ Already in conda environment: $CONDA_PREFIX"
  fi
fi

# --- Auto-detect Architecture ---
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  ARCH="aarch64"
fi
echo "✅ Detected local architecture: $ARCH"

# --- Cleanup Function ---
cleanup() {
  echo "🧹 Cleaning up and restoring files..."
  # Use git to restore the patched files, which is safer than using .bak files
  git restore python/pyproject.toml python/setup.py
  rm -f python/pyvsag/_version.py
  echo "✅ Cleanup complete."
}

# --- Main Build Function ---
run_build() {
  local py_version=$1
  local cibw_build_pattern="cp$(echo "$py_version" | tr -d '.')-*"
  local cibw_version="2.19.1" # Default for modern python

  if [[ "$py_version" == "3.6" ]]; then
    cibw_version="2.11.4"
  fi

  echo "========================================================================"
  echo "🚀 Starting wheel build for Python ${py_version}"
  echo "========================================================================"

  if $HAVE_DOCKER; then
    pip install -q "cibuildwheel==${cibw_version}"
  else
    pip install build
  fi

  # Set trap to call cleanup on exit/error
  trap cleanup EXIT INT TERM

  # Call the reusable script
  bash ./scripts/python/prepare_python_build.sh "$py_version"

  PIP_DEFAULT_TIMEOUT="100" \
  PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  if $HAVE_DOCKER; then
    echo "🛠️  Starting cibuildwheel... "
    CIBW_BUILD="${cibw_build_pattern}" \
    CIBW_ARCHS="${ARCH}" \
    CIBW_TEST_COMMAND="pip install numpy && ls -alF /project/ && python /project/examples/python/example_hnsw.py" \
    cibuildwheel --platform linux --output-dir wheelhouse python
  else 
    echo "🛠️  Starting build..."
    bash scripts/python/build_cpp_for_cibw.sh
    python -m build --wheel --outdir wheelhouse python
    echo "✅ Build complete. Starting test..."
    # Find the most recently created wheel
    LATEST_WHEEL=$(ls -t wheelhouse/pyvsag-*.whl | head -n 1)
    if [ -z "$LATEST_WHEEL" ]; then
        echo "❌ Failed to find the built wheel."
        exit 1
    fi
    echo "   - Installing wheel: ${LATEST_WHEEL}"
    pip install "${LATEST_WHEEL}" --force-reinstall
    echo "   - Running tests..."
    pip install numpy
    python examples/python/example_hnsw.py
  fi

  cleanup
  trap - EXIT INT TERM # Clear the trap

  echo "🎉 Successfully built wheel for Python ${py_version}!"
}

# --- Main Logic ---
mkdir -p wheelhouse
TARGET_VERSION=$1

if [ -z "$TARGET_VERSION" ]; then
  echo "ℹ️  No specific version provided. Building all supported versions: ${SUPPORTED_VERSIONS[*]}"
  for version in "${SUPPORTED_VERSIONS[@]}"; do
    run_build "$version"
  done
else
  if [[ " ${SUPPORTED_VERSIONS[*]} " =~ " ${TARGET_VERSION} " ]]; then
    run_build "$TARGET_VERSION"
  else
    echo "❌ Invalid argument: '$TARGET_VERSION'"
    exit 1
  fi
fi

echo ""
echo "✅ All tasks completed."
echo "📦 Wheels have been generated in the 'wheelhouse' directory:"
ls -l wheelhouse

