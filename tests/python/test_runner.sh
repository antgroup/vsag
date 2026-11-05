#!/bin/bash
# Test runner script for cibuildwheel
# This script runs pytest and analyzes results

set -e

echo "=== Running VSAG Python Tests ==="

# Run pytest with detailed output
python -m pytest /project/tests/python -v --tb=short 2>&1 | tee /tmp/test-output.log

# Analyze results
if grep -q 'FAILED\|ERROR' /tmp/test-output.log; then
    echo "❌ Tests failed!"
    grep -A 5 'FAILED\|ERROR' /tmp/test-output.log || true
    exit 1
elif grep -q 'passed' /tmp/test-output.log; then
    PASSED=$(grep -oP '\d+(?= passed)' /tmp/test-output.log || echo "0")
    echo "✅ All tests passed! ($PASSED tests)"
    exit 0
else
    echo "⚠️  Test results unclear"
    cat /tmp/test-output.log
    exit 1
fi
