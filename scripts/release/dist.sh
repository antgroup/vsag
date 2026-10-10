#!/bin/bash

set -e

# Optional Docker env-file: one unquoted VSAG_THIRDPARTY_<NAME>=URL per line.
# Pinned names are supported; blank lines and full-line comments are ignored.
# Values are literal (including =, &, # and $); no shell expansion or inline comments.
# File values override image defaults; duplicate keys are rejected. Host
# variables are not forwarded. CMake prefers pinned names over legacy names.
thirdparty_env_args=()
if [[ ${VSAG_THIRDPARTY_ENV_FILE+x} ]]; then
    if [[ ! -f "$VSAG_THIRDPARTY_ENV_FILE" || ! -r "$VSAG_THIRDPARTY_ENV_FILE" ]]; then
        echo "VSAG_THIRDPARTY_ENV_FILE must name a readable regular file" >&2
        exit 1
    fi
    # Validate a strict subset of Docker's env-file format before any image build.
    # Reject bare keys (host inheritance), export prefixes, non-ASCII/control bytes,
    # and lines exceeding Docker's scanner limit. Never print file contents.
    LC_ALL=C awk '
        function invalid() {
            printf ("Invalid VSAG_THIRDPARTY_ENV_FILE entry at line %d: " \
                    "expected a unique VSAG_THIRDPARTY_<NAME>=URL\n"), NR > "/dev/stderr"
            exit 1
        }
        {
            sub(/\r$/, "")
            if (length($0) > 65534 || $0 ~ /[^\t -~]/) invalid()
        }
        /^[[:blank:]]*(#|$)/ { next }
        {
            if ($0 !~ /^VSAG_THIRDPARTY_[A-Z0-9_]+=([A-Za-z][A-Za-z0-9+.-]*):\/\/[!-~]+$/) {
                invalid()
            }
            key = substr($0, 1, index($0, "=") - 1)
            if (seen[key]++) invalid()
        }
    ' "$VSAG_THIRDPARTY_ENV_FILE"
    thirdparty_env_args=(--env-file "$VSAG_THIRDPARTY_ENV_FILE")
fi

CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

build() {
    local image_name=$1
    local dockerfile=$2
    local makefile_target=$3
    local dist_name_suffix=$4
    local compile_jobs=${COMPILE_JOBS:-6}

    if ! [[ "$compile_jobs" =~ ^[0-9]+$ ]]; then
        compile_jobs=6
    fi

    docker build -t $image_name -f $dockerfile .

    docker run -u "$CURRENT_UID:$CURRENT_GID" --rm -v "$(pwd):/work" \
           "${thirdparty_env_args[@]}" "$image_name" \
           bash -c "\
           export COMPILE_JOBS=\"$compile_jobs\" && \
           export CMAKE_INSTALL_PREFIX=/tmp/vsag && \
           export EXTRA_DEFINED=\"\${EXTRA_DEFINED:+\$EXTRA_DEFINED }-DVSAG_USE_SYSTEM_DEPS:STRING=OFF\" && \
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
#        export COMPILE_JOBS=6 && \
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
