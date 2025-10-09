#!/bin/bash

set -e

CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

build() {
    local image_name=$1
    local dockerfile=$2
    local makefile_target=$3
    local dist_name_suffix=$4

    docker build -t $image_name -f $dockerfile .

    docker run -u $CURRENT_UID:$CURRENT_GID --rm -v $(pwd):/work $image_name \
           bash -c "\
           export COMPILE_JOBS=48 && \
           export CMAKE_INSTALL_PREFIX=/tmp/vsag && \
           make clean-release && make $makefile_target && make run-dist-tests && make install && \
           mkdir -p ./dist && \
           cp -r /tmp/vsag ./dist/ && \
           cd ./dist && \
           rm -r ./vsag/lib && mv ./vsag/lib64 ./vsag/lib && \
           tar czvf vsag.tar.gz ./vsag && rm -r ./vsag
    "
    version=$(git describe --tags --always --dirty --match "v*")
    dist_name="vsag-$version-$dist_name_suffix"
    mv dist/vsag.tar.gz dist/$dist_name
}

build "vsag-builder-pre-cxx11" \
      "docker/Dockerfile.dist_pre_cxx11_x86" \
      "dist-pre-cxx11-abi" \
      "pre-cxx11-abi.tar.gz"

build "vsag-builder-cxx11" \
      "docker/Dockerfile.dist_cxx11_x86" \
      "dist-cxx11-abi" \
      "cxx11-abi.tar.gz"

# FIXME(wxyu): libcxx deps on clang17, but it cannot install via yum directly
# libcxx version
# docker run -u $CURRENT_UID:$CURRENT_GID --rm -v $(pwd):/work vsag-builder \
#        bash -c "\
#        export COMPILE_JOBS=48 && \
#        export CMAKE_INSTALL_PREFIX=/tmp/vsag && \
#        make clean-release && make dist-libcxx && make install && \
#        mkdir -p ./dist && \
#        cp -r /tmp/vsag ./dist/ && \
#        cd ./dist && \
#        rm -r ./vsag/lib && mv ./vsag/lib64 ./vsag/lib && \
#        tar czvf vsag.tar.gz ./vsag && rm -r ./vsag
# "
# version=$(git describe --tags --always --dirty --match "v*")
# dist_name=vsag-$version-libcxx.tar.gz
# mv dist/vsag.tar.gz dist/$dist_name
