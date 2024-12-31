#!/bin/bash

set -e
set -x

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPTS_DIR}/../"

COVERAGE_DIR="${ROOT_DIR}/coverage"
if [ -d "${COVERAGE_DIR}" ]; then
    rm -rf "${COVERAGE_DIR:?}"/*
else
    mkdir -p "${COVERAGE_DIR}"
fi

lcov --gcov-tool ${SCRIPTS_DIR}/gcov_for_clang.sh \
     --rc branch_coverage=1 \
     --rc geninfo_unexecuted_blocks=1 \
     --include "*/vsag/include/*" \
     --include "*/vsag/src/*" \
     --exclude "*/vsag/include/vsag/expected.hpp*" \
     --exclude "*_test.cpp" \
     --exclude "*/vsag/test_*.cpp" \
     --capture \
     --directory . \
     --output-file  "${COVERAGE_DIR}/coverage.info"

pushd "${COVERAGE_DIR}"
coverages=$(ls coverage.info)
if [ ! "$coverages" ];then
    echo "no coverage file"
    exit 0
fi
popd
