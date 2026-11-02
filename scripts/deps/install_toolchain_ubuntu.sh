#!/usr/bin/env bash
# Ubuntu 26.04 CI and Ubuntu 24.04+ development use GCC 13 and LLVM 20.
# Use distribution packages first, with apt.llvm.org as the LLVM 20 fallback.
set -euo pipefail

. /etc/os-release
if [[ "$ID" != ubuntu ]] || [[ "${VERSION_ID%%.*}" -lt 24 ]]; then
    echo 'Automatic toolchain setup requires Ubuntu 24.04+; install GCC 13+ and LLVM 20 manually.' >&2
    exit 1
fi
apt-get update
apt-get install -y ca-certificates curl gnupg
if ! apt-cache show clang-20 >/dev/null 2>&1; then
    key=$(mktemp)
    trap 'rm -f "$key"' EXIT
    curl --fail --show-error --location --retry 3 https://apt.llvm.org/llvm-snapshot.gpg.key -o "$key"
    gpg --batch --yes --dearmor -o /usr/share/keyrings/llvm-20.gpg "$key"
    printf 'deb [signed-by=/usr/share/keyrings/llvm-20.gpg] https://apt.llvm.org/%s/ llvm-toolchain-%s-20 main\n' \
        "$VERSION_CODENAME" "$VERSION_CODENAME" > /etc/apt/sources.list.d/llvm-20.list
    apt-get update
fi
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc-13 g++-13 gfortran-13 clang-20 clang-tools-20 clangd-20 \
    clang-format-20 clang-tidy-20 llvm-20 llvm-20-dev lld-20 libomp-20-dev
echo 'Selecting GCC 13 and LLVM 20 as defaults; replacing matching /usr/local/bin tool links.' >&2
# Keep compiler, LTO wrappers and coverage tools on the same GCC major version.
for tool in gcc g++ gfortran cpp gcc-ar gcc-nm gcc-ranlib gcov gcov-dump gcov-tool; do
    ln -sf "/usr/bin/$tool-13" "/usr/local/bin/$tool"
done
ln -sf /usr/bin/gcc-13 /usr/local/bin/cc
ln -sf /usr/bin/g++-13 /usr/local/bin/c++
for tool in clang clang++ clang-format clang-tidy clangd clang-apply-replacements ld.lld llvm-config; do
    ln -sf "/usr/lib/llvm-20/bin/$tool" "/usr/local/bin/$tool"
done
