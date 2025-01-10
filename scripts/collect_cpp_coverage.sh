#!/bin/bash

set -e
set -x

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPTS_DIR}/../" && pwd )"
PARENT_DIR="$( cd "${ROOT_DIR}/../" && pwd )"

COVERAGE_DIR="${ROOT_DIR}/coverage"
if [ -d "${COVERAGE_DIR}" ]; then
    rm -rf "${COVERAGE_DIR:?}"/*
else
    mkdir -p "${COVERAGE_DIR}"
fi

pushd "${PARENT_DIR}"
lcov --rc branch_coverage=1 \
     --rc geninfo_unexecuted_blocks=1 \
     --parallel 8 \
     --directory vsag \
     --capture \
     --substitute "s#${PARENT_DIR}/##g" \
     --ignore-errors mismatch,mismatch \
     --ignore-errors count,count \
     --output-file ${COVERAGE_DIR}/coverage.info
lcov --remove ${COVERAGE_DIR}/coverage.info \
     '/usr/*' \
     'vsag/build/*' \
     'vsag/tests/*' \
     --ignore-errors inconsistent,inconsistent \
     --output-file ${COVERAGE_DIR}/coverage.info
lcov --list ${COVERAGE_DIR}/coverage.info \
     --ignore-errors inconsistent,inconsistent
popd

pushd "${COVERAGE_DIR}"
coverages=$(ls coverage.info)
if [ ! "$coverages" ];then
    echo "no coverage file"
    exit 0
fi
popd
