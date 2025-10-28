#!/bin/bash
set -eo pipefail

CMAKE_BUILD_DIR="./build-release/"
PYTHON_EXECUTABLE="${PYTHON:-$(which python3 || which python)}"
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    echo "Error: Python executable not found or not executable: $PYTHON_EXECUTABLE"
    exit 1
fi

echo "=== Python Configuration ==="
echo "Python executable: $PYTHON_EXECUTABLE"
$PYTHON_EXECUTABLE --version
PYTHON_VERSION=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_NO_DOT=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PYTHON_SOABI=$($PYTHON_EXECUTABLE -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))")
EXPECTED_SUFFIX=$($PYTHON_EXECUTABLE -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Python version: $PYTHON_VERSION"
echo "Python SOABI: $PYTHON_SOABI"
echo "Expected suffix: $EXPECTED_SUFFIX"

echo "Cleaning previous extension modules in python/pyvsag/..."
rm -rf python/pyvsag/*.so python/pyvsag/*.so.*

echo "=== Configuring CMake ==="
cmake -S. -B$CMAKE_BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -s" \
    -DENABLE_PYBINDS=ON \
    -DENABLE_TESTS=OFF \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython3_ROOT_DIR="$(dirname $(dirname $PYTHON_EXECUTABLE))" \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DMKL_STATIC_LINK=ON

echo "=== Building project ==="
cmake --build $CMAKE_BUILD_DIR --parallel $(nproc || sysctl -n hw.ncpu)

echo "=== Looking for generated .so files ==="
echo "Expected pattern: _pyvsag*${PYTHON_SOABI}*.so or _pyvsag.cpython-${PYTHON_VERSION_NO_DOT}*.so"

PYVSAG_CPYTHON_SO=""
for pattern in "_pyvsag*${PYTHON_SOABI}*.so" "_pyvsag.cpython-${PYTHON_VERSION_NO_DOT}*.so" "_pyvsag*.so"; do
    echo "Searching for: $pattern"
    FOUND=$(find $CMAKE_BUILD_DIR -name "$pattern" -type f 2>/dev/null | head -n1)
    if [ -n "$FOUND" ]; then
        PYVSAG_CPYTHON_SO="$FOUND"
        echo "Found: $PYVSAG_CPYTHON_SO"
        break
    fi
done

if [ -z "$PYVSAG_CPYTHON_SO" ]; then
    echo "Error: Could not find the compiled _pyvsag*.so file."
    echo "Build directory contents:"
    find $CMAKE_BUILD_DIR -name "*.so" -o -name "*.so.*" 2>/dev/null || echo "No .so files found"
    
    echo ""
    echo "CMake Python configuration:"
    grep -i python $CMAKE_BUILD_DIR/CMakeCache.txt | head -20 || true
    exit 1
fi

SO_BASENAME=$(basename "$PYVSAG_CPYTHON_SO")
echo "Generated .so file: $SO_BASENAME"

if [[ "$SO_BASENAME" =~ cpython-${PYTHON_VERSION_NO_DOT} ]] || [[ "$SO_BASENAME" =~ ${PYTHON_SOABI} ]]; then
    echo "✓ Version match confirmed"
else
    echo "⚠ WARNING: .so file may not match Python version!"
    echo "  Expected: cpython-${PYTHON_VERSION_NO_DOT} or ${PYTHON_SOABI}"
    echo "  Got: $SO_BASENAME"
    
    if [[ "$SO_BASENAME" =~ cpython-([0-9]+)- ]]; then
        OLD_VERSION="${BASH_REMATCH[1]}"
        if [ "$OLD_VERSION" != "${PYTHON_VERSION_NO_DOT}" ]; then
            echo "  Detected version mismatch: $OLD_VERSION vs ${PYTHON_VERSION_NO_DOT}"
        fi
    fi
fi

echo "Original size of $(basename $PYVSAG_CPYTHON_SO): $(du -h $PYVSAG_CPYTHON_SO | cut -f1)"
strip --strip-unneeded "$PYVSAG_CPYTHON_SO" || echo "Strip failed, continuing anyway"
echo "Stripped size of $(basename $PYVSAG_CPYTHON_SO): $(du -h $PYVSAG_CPYTHON_SO | cut -f1)"

echo "Copying to: python/pyvsag/"
mkdir -p python/pyvsag/
cp "$PYVSAG_CPYTHON_SO" python/pyvsag/

LIBVSAG_SO=$(find $CMAKE_BUILD_DIR -name "libvsag.so*" -type f | head -n1)

if [ -n "$LIBVSAG_SO" ]; then
    echo "Found libvsag.so at: $LIBVSAG_SO"
    
    if ldd "$PYVSAG_CPYTHON_SO" 2>/dev/null | grep -q "libvsag.so.*=> not found"; then
        echo "libvsag.so not in standard path, copying to package"
        strip --strip-unneeded "$LIBVSAG_SO" || true
        cp "$LIBVSAG_SO" python/pyvsag/
    else
        strip --strip-unneeded "$LIBVSAG_SO" || true
        echo "libvsag.so is accessible, letting auditwheel handle it"
    fi
fi

echo "Stripping all SO files in build directory..."
find $CMAKE_BUILD_DIR -name "*.so*" -type f -exec sh -c '
    for file; do
        strip --strip-unneeded "$file" 2>/dev/null || true
    done
' sh {} +

if [ -d "$CMAKE_BUILD_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$CMAKE_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

echo ""
echo "=== Build Summary ==="
echo "Python: $PYTHON_EXECUTABLE (version $PYTHON_VERSION)"
echo "Built: $(basename $PYVSAG_CPYTHON_SO)"
echo "Location: python/pyvsag/"

echo ""
echo "=== Final package contents ==="
ls -lh python/pyvsag/*.so 2>/dev/null || echo "No .so files in package"

echo ""
echo "Build script completed successfully!"
