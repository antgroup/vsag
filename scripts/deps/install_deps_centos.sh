# install python3-devel
yum install -y gcc gcc-c++ gcc-gfortran python3-devel libaio-devel libcurl-devel

ARCH=$(uname -m)

if [[ "${ARCH}" == "x86_64" ]]; then
    # install openmp for alibaba clang 11 (only available on x86_64)
    yum install -b current -y libomp11-devel libomp11

    # install Intel MKL (only available on x86_64)
    yum install -y ca-certificates
    yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    yum install -y intel-mkl-64bit-2020.0-088
else
    # MKL is not supported on aarch64. We fallback to openblas/generic openmp.
    yum install -y ca-certificates libgomp
fi
