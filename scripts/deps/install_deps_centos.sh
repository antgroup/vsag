set -e

# install python3-devel
yum install -y epel-release && yum install -y gcc gcc-c++ gcc-gfortran python3-devel libaio-devel libcurl-devel ca-certificates openblas-devel lapack-devel

yum install -y libgomp
