#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# --- Configuration ---
LINUX_SUPPORTED_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12" "3.13" "3.14")
MACOS_SUPPORTED_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")
HAVE_DOCKER=false

# --- Helper Functions ---
join_versions() {
  local IFS=", "
  echo "$*"
}

version_is_supported() {
  local py_version=$1
  shift
  local version
  for version in "$@"; do
    if [[ "${version}" == "${py_version}" ]]; then
      return 0
    fi
  done
  return 1
}

python_matches_version() {
  local python_cmd=$1
  local py_version=$2
  "${python_cmd}" -c "
import sys
expected = tuple(map(int, '${py_version}'.split('.')))
sys.exit(0 if sys.version_info[:2] == expected else 1)
" >/dev/null 2>&1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    echo "   - Installing uv with Homebrew..." >&2
    brew install uv
    command -v uv
    return 0
  fi

  echo "   - Installing uv to manage missing local Python versions..." >&2
  python3 -m pip install --user --timeout 60 -q uv

  local uv_bin
  uv_bin=$(python3 - <<'PY'
import os
import site

print(os.path.join(site.USER_BASE, "bin", "uv"))
PY
)
  if [[ -x "${uv_bin}" ]]; then
    echo "${uv_bin}"
    return 0
  fi

  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  echo "❌ Failed to install uv. Add Python's user base bin directory to PATH and retry." >&2
  exit 1
}

resolve_python() {
  local py_version=$1
  local candidate
  local candidates=(
    "python${py_version}"
    "/opt/homebrew/bin/python${py_version}"
    "/usr/local/bin/python${py_version}"
    "/Library/Frameworks/Python.framework/Versions/${py_version}/bin/python${py_version}"
  )

  for candidate in "${candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1 &&
        python_matches_version "${candidate}" "${py_version}"; then
      command -v "${candidate}"
      return 0
    fi
  done

  if [[ "$(uname -s)" == "Darwin" ]]; then
    local uv_bin
    uv_bin=$(ensure_uv)
    echo "   - Python ${py_version} not found locally; installing with uv..." >&2
    "${uv_bin}" python install "${py_version}"
    "${uv_bin}" python find "${py_version}"
    return 0
  fi

  echo "❌ Python ${py_version} was not found. Install it or choose another PY_VERSION." >&2
  exit 1
}

activate_target_venv() {
  local python_cmd=$1
  local py_version=$2
  local venv_dir=".venv-py${py_version//./}"

  if [[ ! -x "${venv_dir}/bin/python" ]] ||
      ! python_matches_version "${venv_dir}/bin/python" "${py_version}"; then
    rm -rf "${venv_dir}"
    echo "   - Creating virtual environment at ./${venv_dir}..."
    "${python_cmd}" -m venv "${venv_dir}"
  fi

  # shellcheck disable=SC1090
  source "${venv_dir}/bin/activate"
  python -m pip install -q --upgrade pip
}

cleanup() {
  echo "🧹 Cleaning up..."
  rm -f python/pyvsag/_version.py
  rm -rf python/build python/*.so python/pyvsag/*.so
  echo "✅ Cleanup complete."
}

test_latest_wheel() {
  local py_version=$1
  local py_tag="cp${py_version//./}"
  local latest_wheel

  latest_wheel=$(ls -t wheelhouse/pyvsag-*${py_tag}*.whl 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_wheel}" ]]; then
    echo "❌ Failed to find a built wheel for Python ${py_version}."
    exit 1
  fi

  echo "   - Installing wheel: ${latest_wheel}"
  python -m pip install "${latest_wheel}" --force-reinstall
  echo "   - Running tests..."
  python -m pip install -q numpy pytest
  python tests/python/run_test.py
}

run_macos_build() {
  local py_version=$1
  local python_cmd
  local deployment_target=${MACOSX_DEPLOYMENT_TARGET:-11.0}

  echo "========================================================================"
  echo "🚀 Starting macOS wheel build for Python ${py_version}"
  echo "========================================================================"

  bash ./scripts/deps/install_deps_macos.sh
  python_cmd=$(resolve_python "${py_version}")
  activate_target_venv "${python_cmd}" "${py_version}"
  python -m pip install -q build

  trap cleanup EXIT INT TERM
  rm -rf python/build python/*.so python/pyvsag/*.so
  rm -f wheelhouse/pyvsag-*cp${py_version//./}*.whl
  bash ./scripts/python/prepare_python_build.sh "${py_version}"

  echo "🛠️  Starting local macOS build..."
  echo "   - MACOSX_DEPLOYMENT_TARGET=${deployment_target}"

  local build_env=(
    "VSAG_USE_SYSTEM_DEPS=${VSAG_USE_SYSTEM_DEPS:-OFF}"
    "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(sysctl -n hw.ncpu)}"
    "MACOSX_DEPLOYMENT_TARGET=${deployment_target}"
  )
  env "${build_env[@]}" python -m build --wheel --outdir wheelhouse python

  echo "✅ Build complete. Starting test..."
  test_latest_wheel "${py_version}"
  cleanup
  trap - EXIT INT TERM

  echo "🎉 Successfully built macOS wheel for Python ${py_version}!"
}

run_linux_build() {
  local py_version=$1
  local cibw_build_pattern="cp$(echo "${py_version}" | tr -d '.')-*"
  local cibw_version="3.3.1"
  local use_uvx=true
  local arch

  arch=$(uname -m)
  if [[ "${arch}" == "arm64" ]]; then
    arch="aarch64"
  fi

  if docker info >/dev/null 2>&1; then
    HAVE_DOCKER=true
  else
    HAVE_DOCKER=false
  fi

  if [[ "${py_version}" == "3.6" ]]; then
    cibw_version="2.11.4"
    use_uvx=false
  elif [[ "${py_version}" == "3.7" ]]; then
    cibw_version="2.23.3"
    use_uvx=false
  elif python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
      2>/dev/null; then
    use_uvx=false
  fi

  echo "========================================================================"
  echo "🚀 Starting Linux wheel build for Python ${py_version}"
  echo "========================================================================"

  trap cleanup EXIT INT TERM
  rm -rf python/build python/*.so python/pyvsag/*.so
  rm -f wheelhouse/pyvsag-*cp${py_version//./}*.whl
  bash ./scripts/python/prepare_python_build.sh "${py_version}"

  if ${HAVE_DOCKER}; then
    echo "🛠️  Starting cibuildwheel..."
    if ${use_uvx}; then
      python3 -m pip install -q uv 2>/dev/null || true
      CIBW_BUILD="${cibw_build_pattern}" \
      CIBW_ARCHS="${arch}" \
      CIBW_TEST_COMMAND="pip install numpy pytest && python /project/tests/python/run_test.py" \
        uvx --python 3.12 --from "cibuildwheel==${cibw_version}" \
        cibuildwheel --platform linux --output-dir wheelhouse python
    else
      python3 -m pip install -q "cibuildwheel==${cibw_version}"
      CIBW_BUILD="${cibw_build_pattern}" \
      CIBW_ARCHS="${arch}" \
      CIBW_TEST_COMMAND="pip install numpy pytest && python /project/tests/python/run_test.py" \
        cibuildwheel --platform linux --output-dir wheelhouse python
    fi
  else
    echo "⚠️ Docker daemon is not running. Building a local Linux wheel without Docker."
    local python_cmd
    python_cmd=$(resolve_python "${py_version}")
    activate_target_venv "${python_cmd}" "${py_version}"
    python -m pip install -q build
    python -m build --wheel --outdir wheelhouse python
    test_latest_wheel "${py_version}"
  fi

  cleanup
  trap - EXIT INT TERM

  echo "🎉 Successfully built Linux wheel for Python ${py_version}!"
}

run_build() {
  local py_version=$1
  if [[ "$(uname -s)" == "Darwin" ]]; then
    run_macos_build "${py_version}"
  else
    run_linux_build "${py_version}"
  fi
}

# --- Prerequisite Checks ---
echo "🔎 Checking prerequisites..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ 'python3' command not found. Please ensure Python 3 is installed."
  exit 1
fi
if [[ ! -f "scripts/python/prepare_python_build.sh" ]]; then
  echo "❌ Preparation script not found at 'scripts/python/prepare_python_build.sh'."
  exit 1
fi
echo "✅ Prerequisites met."

# --- Main Logic ---
mkdir -p wheelhouse
TARGET_VERSION=${1:-}

if [[ "$(uname -s)" == "Darwin" ]]; then
  SUPPORTED_VERSIONS=("${MACOS_SUPPORTED_VERSIONS[@]}")
else
  SUPPORTED_VERSIONS=("${LINUX_SUPPORTED_VERSIONS[@]}")
fi

if [[ -z "${TARGET_VERSION}" ]]; then
  supported_versions=$(join_versions "${SUPPORTED_VERSIONS[@]}")
  echo "ℹ️  No specific version provided."
  echo "   - Building all supported versions: ${supported_versions}"
  for version in "${SUPPORTED_VERSIONS[@]}"; do
    run_build "${version}"
  done
else
  if version_is_supported "${TARGET_VERSION}" "${SUPPORTED_VERSIONS[@]}"; then
    run_build "${TARGET_VERSION}"
  else
    echo "❌ Invalid argument: '${TARGET_VERSION}'"
    echo "Supported versions on this platform: $(join_versions "${SUPPORTED_VERSIONS[@]}")"
    exit 1
  fi
fi

echo ""
echo "✅ All tasks completed."
echo "📦 Wheels have been generated in the 'wheelhouse' directory:"
ls -l wheelhouse
