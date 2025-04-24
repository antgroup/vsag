#!/bin/bash

set -e

CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

docker build -t vsag-builder -f docker/Dockerfile.dist_x86 .

# old-abi version
docker run -u $CURRENT_UID:$CURRENT_GID --rm -v $(pwd):/work vsag-builder \
       bash -c "\
       export COMPILE_JOBS=48 && \
       export CMAKE_INSTALL_PREFIX=/tmp/vsag && \
       make clean-release && make dist-old-abi && make install && \
       mkdir -p ./dist && \
       cp -r /tmp/vsag ./dist/ && \
       cd ./dist && \
       rm -r ./vsag/lib && mv ./vsag/lib64 ./vsag/lib && \
       tar czvf vsag.tar.gz ./vsag && rm -r ./vsag
"
version=$(git describe --tags --always --dirty --match "v*")
dist_name=vsag-$version-old-abi.tar.gz
mv dist/vsag.tar.gz dist/$dist_name

# cxx11-abi version
docker run -u $CURRENT_UID:$CURRENT_GID --rm -v $(pwd):/work vsag-builder \
       bash -c "\
       export COMPILE_JOBS=48 && \
       export CMAKE_INSTALL_PREFIX=/tmp/vsag && \
       make clean-release && make dist-cxx11-abi && make install && \
       mkdir -p ./dist && \
       cp -r /tmp/vsag ./dist/ && \
       cd ./dist && \
       rm -r ./vsag/lib && mv ./vsag/lib64 ./vsag/lib && \
       tar czvf vsag.tar.gz ./vsag && rm -r ./vsag
"
version=$(git describe --tags --always --dirty --match "v*")
dist_name=vsag-$version-cxx11-abi.tar.gz
mv dist/vsag.tar.gz dist/$dist_name

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
