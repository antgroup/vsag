#!/bin/bash

set -eo pipefail

MIN_LINE_COVERAGE=90

line_coverage=$(lcov --rc branch_coverage=1 \
  --summary coverage/coverage.info \
  --ignore-errors inconsistent,inconsistent | \
  grep "lines......" | \
  awk '/lines/ { print $2 }' | \
  cut -d '%' -f 1)

if awk -v cov="${line_coverage}" -v min="${MIN_LINE_COVERAGE}" 'BEGIN { exit !(cov+0 >= min+0) }'; then
  echo "line coverage is ${line_coverage}, meets ${MIN_LINE_COVERAGE}"
  exit 0
else
  echo "line coverage is ${line_coverage}, less than ${MIN_LINE_COVERAGE}"
  exit 1
fi
