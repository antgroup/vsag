#!/bin/bash

set -e
set -x

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPTS_DIR}/../../" && pwd )"

COVERAGE_DIR="${ROOT_DIR}/coverage"
COVERAGE_RAW_FILE="${COVERAGE_DIR}/coverage.raw.info"
COVERAGE_SCOPED_FILE="${COVERAGE_DIR}/coverage.scoped.info"
COVERAGE_FILE="${COVERAGE_DIR}/coverage.info"
if [ -d "${COVERAGE_DIR}" ]; then
    rm -rf "${COVERAGE_DIR:?}"/*
else
    mkdir -p "${COVERAGE_DIR}"
fi

# Limit capture to repository-owned production sources before lcov runs consistency checks.
# The later extract is a defense-in-depth check and normalizes paths for local and Codecov use.
lcov --rc branch_coverage=1 \
     --rc geninfo_unexecuted_blocks=1 \
     --parallel 8 \
     --directory "${ROOT_DIR}" \
     --capture \
     --exclude '/usr/*' \
     --include "${ROOT_DIR}/src/*" \
     --include "${ROOT_DIR}/include/*" \
     --ignore-errors mismatch,mismatch \
     --ignore-errors count,count \
     --output-file "${COVERAGE_RAW_FILE}"

# These two files are not maintained production sources: version.h is generated at build time,
# and expected.hpp is a vendored public compatibility header.
lcov --rc branch_coverage=1 \
     --remove "${COVERAGE_RAW_FILE}" \
     "${ROOT_DIR}/src/version.h" \
     "${ROOT_DIR}/include/vsag/expected.hpp" \
     --ignore-errors inconsistent,inconsistent \
     --ignore-errors unused,unused \
     --output-file "${COVERAGE_SCOPED_FILE}"

lcov --rc branch_coverage=1 \
     --extract "${COVERAGE_SCOPED_FILE}" \
     'src/*' \
     'include/*' \
     --substitute "s#${ROOT_DIR}/##g" \
     --ignore-errors inconsistent,inconsistent \
     --ignore-errors unused,unused \
     --output-file "${COVERAGE_FILE}"

rm -f "${COVERAGE_RAW_FILE}" "${COVERAGE_SCOPED_FILE}"

python3 "${SCRIPTS_DIR}/verify_cpp_coverage.py" trace "${COVERAGE_FILE}"

lcov --rc branch_coverage=1 \
     --list "${COVERAGE_FILE}" \
     --ignore-errors inconsistent,inconsistent,child
