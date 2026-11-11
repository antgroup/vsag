#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR=".venv"
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
  echo "🧹 Cleaning up..."
  rm -f python/pyvsag/_version.py
  rm -rf python/build python/*.so python/pyvsag/*.so
  echo "✅ Cleanup complete."
}

# --- Main Build Function ---
run_build() {
  echo "========================================================================"
  echo "🚀 Starting pyvsag wheel build"
  echo "========================================================================"

  if $HAVE_DOCKER; then
    pip install -q cibuildwheel
  else
    pip install build
  fi

  trap cleanup EXIT INT TERM

  rm -rf python/build python/*.so python/pyvsag/*.so
  bash ./scripts/python/prepare_python_build.sh

  PIP_DEFAULT_TIMEOUT="100"
  if $HAVE_DOCKER; then
    echo "🛠️  Starting cibuildwheel... "
    CIBW_ARCHS="${ARCH}" \
    CIBW_TEST_COMMAND="pip install numpy pytest && python /project/tests/python/run_test.py" \
    cibuildwheel --platform linux --output-dir wheelhouse python
  else
    echo "🛠️  Starting build..."
    python -m build --wheel --outdir wheelhouse python
    echo "✅ Build complete. Starting test..."
    LATEST_WHEEL=$(ls -t wheelhouse/pyvsag-*.whl | head -n 1)
    if [ -z "$LATEST_WHEEL" ]; then
        echo "❌ Failed to find the built wheel."
        exit 1
    fi
    echo "   - Installing wheel: ${LATEST_WHEEL}"
    pip install "${LATEST_WHEEL}" --force-reinstall
    echo "   - Running tests..."
    pip install pytest
    cd tests/python && python -m pytest . -v --tb=short 2>&1 | tee test-output.log || (cat test-output.log && exit 1)
    cd ../..
  fi

  cleanup
  trap - EXIT INT TERM
  echo "🎉 Successfully built the pyvsag wheel!"
}

mkdir -p wheelhouse
run_build

echo ""
echo "📦 Wheels have been generated in the 'wheelhouse' directory:"
ls -l wheelhouse
