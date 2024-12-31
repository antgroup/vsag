#!/bin/bash

line_coverage=`lcov --summary coverage/coverage.info | grep "lines......" | awk '/lines/ { print $2 }' | cut -d '%' -f 1`
line_coverage=$(printf "%.0f" $line_coverage)
if [ "$line_coverage" -gt 84 ]; then
  echo "line coverage is ${line_coverage}, more than 84"
  exit 0;
else
  echo "line coverage is ${line_coverage}, less than 84"
  exit 1;
fi
