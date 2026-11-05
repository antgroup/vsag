#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# This script prepares the project for building a Python wheel.
# It handles version generation and config file patching based on the Python version.

# --- Input Validation ---
if [ -z "$1" ]; then
  echo "❌ Error: Python version must be provided as the first argument."
  echo "Usage: $0 <python_version>"
  exit 1
fi

PY_VERSION=$1
echo "🚀 Preparing build for Python ${PY_VERSION}..."

# --- Main Logic ---

# 1. Statically generate the version file using setuptools_scm
echo "   - Installing setuptools_scm to generate version..."
pip install -q setuptools_scm

echo "   - Generating python/pyvsag/_version.py..."
python3 -c "
import setuptools_scm
try:
    # Explicitly configure setuptools_scm since pyproject.toml is in a subdir
    version = setuptools_scm.get_version(
        root='.',
        version_scheme='release-branch-semver',
        local_scheme='no-local-version'
    )
except Exception:
    version = '0.0.0'

with open('python/pyvsag/_version.py', 'w') as f:
    f.write(f'__version__ = \"{version}\"\\n')
print(f'   - Version generated: {version}')
"

# 2. Done
echo "✅ Build preparation complete for Python ${PY_VERSION}."
