arch=$(uname -m)
common_pkgs=(
    ccache
    cmake
    g++
    gcc
    gfortran
    lcov
    libaio-dev
    libcurl4-openssl-dev
    liblapacke-dev
    libomp-15-dev
    libopenblas-dev
    liburing-dev
    make
    ninja-build
    python3-dev
)
install_intel_mkl="${VSAG_INSTALL_INTEL_MKL:-ON}"
arch_pkgs=()

if [[ "$arch" == "x86_64" ]]; then
    echo "Executing apt install for x86_64"
    # OpenBLAS-only CI jobs skip the unrelated MKL package to reduce setup time.
    if [[ "$install_intel_mkl" == "ON" ]]; then
        arch_pkgs+=(intel-mkl)
    elif [[ "$install_intel_mkl" == "OFF" ]]; then
        :
    else
        echo "VSAG_INSTALL_INTEL_MKL must be ON or OFF: $install_intel_mkl"
        exit 1
    fi
    apt update &&
        DEBIAN_FRONTEND=noninteractive apt install -y "${common_pkgs[@]}" "${arch_pkgs[@]}"
elif [[ "$arch" == "aarch64" ]]; then
    echo "Executing apt install for aarch64"
    apt update && DEBIAN_FRONTEND=noninteractive apt install -y "${common_pkgs[@]}"
else
    echo "Unknown architecture: $arch"
    exit 1
fi
