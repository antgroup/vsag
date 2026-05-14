# install python3-devel
yum install -y gfortran python3-devel libaio-devel libcurl-devel ca-certificates

# install openmp for alibaba clang 11
yum install -b current -y libomp11-devel libomp11

yum install -y libgomp
