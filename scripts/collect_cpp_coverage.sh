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

lcov --rc branch_coverage=1 \
     --rc geninfo_unexecuted_blocks=1 \
     --parallel 8 \
     --directory . \
     --capture \
     --ignore-errors mismatch,mismatch \
     --ignore-errors count,count \
     --output-file ${COVERAGE_DIR}/coverage.info

lcov --remove ${COVERAGE_DIR}/coverage.info '/usr/*' '*/build/*' '*/vsag/tests/*' --ignore-errors inconsistent,inconsistent --output-file ${COVERAGE_DIR}/coverage.info
lcov --list ${COVERAGE_DIR}/coverage.info --ignore-errors inconsistent,inconsistent

pushd "${COVERAGE_DIR}"
coverages=$(ls coverage.info)
if [ ! "$coverages" ];then
    echo "no coverage file"
    exit 0
fi
popd
