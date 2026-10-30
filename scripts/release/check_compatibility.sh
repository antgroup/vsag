#!/usr/bin/env bash
# Validate the installed package, never the source/build-tree include paths.
set -euo pipefail

prefix=$(cd "${1:?usage: check_compatibility.sh INSTALL_PREFIX ABI [REPORT_DIR]}" && pwd)
abi=${2:?ABI must be 0 or 1}
[[ "$abi" =~ ^[01]$ ]] || { echo 'ABI must be 0 or 1' >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd "$script_dir/../../tests/compatibility" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
report_dir=${3:-$work/reports}
mkdir -p "$report_dir"
cxx=${CXX:-c++}

# Do not let developer include/search paths mask a broken installed package.
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
find "$prefix/include/vsag" -type f \( -name '*.h' -o -name '*.hpp' \) -print0 > "$work/headers"
[[ -s "$work/headers" ]] || { echo 'No public headers found in installed package' >&2; exit 1; }
while IFS= read -r -d '' header; do
    relative=${header#"$prefix/include/"}
    printf '#include <%s>\n' "$relative" > "$work/header.cpp"
    "$cxx" -std=c++17 -pedantic-errors -D_GLIBCXX_USE_CXX11_ABI="$abi" \
        -I"$prefix/include" -fsyntax-only "$work/header.cpp"
done < "$work/headers"

config=$(find "$prefix" -name vsagConfig.cmake -print -quit)
[[ -n "$config" ]] || { echo 'Installed CMake package missing' >&2; exit 1; }
cmake -S "$source_dir" -B "$work/client" \
    -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_PREFIX_PATH="$prefix" \
    -Dvsag_DIR:PATH="$(dirname "$config")" \
    -DVSAG_CLIENT_CXX11_ABI="$abi"
cmake --build "$work/client" --parallel 2
(cd "$work/client" && ctest --output-on-failure)
cp "$work/client/compile_commands.json" "$report_dir/client-compile-commands.json"

library=$(cat "$work/client/installed-library.txt")
[[ -f "$library" ]] || { echo 'Installed shared library missing' >&2; exit 1; }
readelf --version-info "$library" > "$report_dir/symbol-versions.txt"
readelf -d "$library" > "$report_dir/dynamic.txt"
ldd "$library" > "$report_dir/dependencies.txt"
ldd "$work/client/vsag_cxx17_client" > "$report_dir/client-dependencies.txt"
if grep -q 'not found' "$report_dir/"*dependencies.txt; then
    echo 'Unresolved dynamic dependency' >&2; exit 1
fi
# Absolute NEEDED entries and build paths in RUNPATH/RPATH are not redistributable.
if grep -E '\(NEEDED\).*\[/|\((RPATH|RUNPATH)\).*\[([^]]*:)?/' "$report_dir/dynamic.txt"; then
    echo 'Build-machine path leaked into ELF dependencies' >&2; exit 1
fi
# Release callers provide their existing manylinux glibc ceiling (2.17 or 2.28).
if [[ -n "${VSAG_MAX_GLIBC:-}" ]]; then
    while read -r required; do
        highest=$(printf '%s\n%s\n' "$VSAG_MAX_GLIBC" "$required" | sort -V | tail -1)
        if [[ "$highest" != "$VSAG_MAX_GLIBC" ]]; then
            echo "GLIBC_$required exceeds release baseline $VSAG_MAX_GLIBC" >&2; exit 1
        fi
    done < <(grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' "$report_dir/symbol-versions.txt" | sed 's/GLIBC_//' | sort -Vu)
fi
# Minimum-runtime execution remains a separate gate.
echo "Installed C++17/ABI=$abi compatibility passed; ELF reports: $report_dir"
