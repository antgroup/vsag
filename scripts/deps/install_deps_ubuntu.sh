arch=$(uname -m)
common_pkgs="gfortran python3-dev libomp-15-dev gcc make cmake g++ lcov libaio-dev libopenblas-dev liblapacke-dev libcurl4-openssl-dev liburing-dev"

if [[ "$arch" == "x86_64" ]]; then
    echo "Executing apt install for x86_64"
    apt update && DEBIAN_FRONTEND=noninteractive apt install -y $common_pkgs intel-mkl
elif [[ "$arch" == "aarch64" ]]; then
    echo "Executing apt install for aarch64"
    apt update && DEBIAN_FRONTEND=noninteractive apt install -y $common_pkgs
else
    echo "Unknown architecture: $arch"
    exit 1
fi
