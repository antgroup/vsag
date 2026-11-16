#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -eo pipefail

# --- Configuration ---
VENV_DIR=".venv"
HAVE_DOCKER=true
WHEEL_BUILD_DIR=""

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
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "🍎 Building a native macOS wheel."
  HAVE_DOCKER=false
elif ! docker info > /dev/null 2>&1; then
  echo "⚠️ Docker daemon is not running. Build without Docker."
  HAVE_DOCKER=false
fi
echo "✅ Prerequisites met."

# --- Virtual Environment Setup ---
echo "🔎 Setting up Python virtual environment..."
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
  # Not in any virtual environment, create and activate .venv
  if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    if [ -e "$VENV_DIR" ] && [ ! -f "${VENV_DIR}/pyvenv.cfg" ] && [ -n "$(ls -A "$VENV_DIR" 2>/dev/null)" ]; then
      echo "❌ './${VENV_DIR}' exists but is not a Python virtual environment."
      echo "   Remove it, or activate your own virtual environment and run the build again."
      exit 1
    fi
    echo "   - Virtual environment not found or incomplete, creating at './${VENV_DIR}'..."
    python3 -m venv --clear "$VENV_DIR" || {
      echo "❌ Could not create a virtual environment. Install the venv module (for example"
      echo "   'apt install python3-venv') or activate your own virtual environment first."
      exit 1
    }
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
  rm -rf python/build python/*.so python/pyvsag/*.so python/pyvsag/*.dylib
  if [ -n "$WHEEL_BUILD_DIR" ]; then
    rm -rf "$WHEEL_BUILD_DIR"
  fi
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
    if [[ "$(uname -s)" == "Darwin" ]]; then
      pip install delocate
    fi
  fi

  trap cleanup EXIT INT TERM

  rm -rf python/build python/*.so python/pyvsag/*.so python/pyvsag/*.dylib
  bash ./scripts/python/prepare_python_build.sh

  PIP_DEFAULT_TIMEOUT="100"
  if $HAVE_DOCKER; then
    echo "🛠️  Starting cibuildwheel... "
    CIBW_ARCHS="${ARCH}" \
    CIBW_ENVIRONMENT_PASS_LINUX="CMAKE_BUILD_PARALLEL_LEVEL" \
    CIBW_TEST_COMMAND="pip install numpy pytest && python /project/tests/python/run_test.py" \
    cibuildwheel --platform linux --output-dir wheelhouse python
  else
    echo "🛠️  Starting build..."
    mkdir -p wheelhouse
    WHEEL_BUILD_DIR=$(mktemp -d "$PWD/wheelhouse/.native-build.XXXXXX")
    python -m build --wheel --outdir "$WHEEL_BUILD_DIR" python
    wheels=("$WHEEL_BUILD_DIR"/pyvsag-*.whl)
    if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
        echo "❌ Expected exactly one newly built wheel."
        exit 1
    fi
    if [[ "$(uname -s)" == "Darwin" ]]; then
      # Repair any remaining dynamic dependencies and validate the shipped architecture.
      delocate-wheel --require-archs "$(uname -m)" \
        --wheel-dir "$WHEEL_BUILD_DIR/repaired" "${wheels[0]}"
      wheels=("$WHEEL_BUILD_DIR"/repaired/pyvsag-*.whl)
      python scripts/python/check_macos_wheel.py "${wheels[0]}"
    fi
    LATEST_WHEEL="wheelhouse/$(basename "${wheels[0]}")"
    cp "${wheels[0]}" "$LATEST_WHEEL"
    echo "✅ Build complete. Starting test..."
    echo "   - Installing wheel: ${LATEST_WHEEL}"
    pip install "${LATEST_WHEEL}" --force-reinstall
    echo "   - Running tests..."
    pip install pytest
    (cd tests/python && python -m pytest . -v --tb=short 2>&1 | tee test-output.log)
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
