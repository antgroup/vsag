arch=$(uname -m)

if [[ "$arch" == "x86_64" ]]; then
    echo "Executing apt install for x86_64"
    apt update && apt install -y gfortran python3-dev libomp-15-dev gcc make cmake g++ lcov libaio-dev intel-mkl libcurl4-openssl-dev
elif [[ "$arch" == "aarch64" ]]; then
    echo "Executing apt install for aarch64"
    apt update && apt install -y gfortran python3-dev libomp-15-dev gcc make cmake g++ lcov libaio-dev libcurl4-openssl-dev
else
    echo "Unknown architecture: $arch"
fi
