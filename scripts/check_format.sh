#!/usr/bin/bash

make fmt
if [[ -n $(git status --porcelain) ]]; then
    echo ""
    echo "code format issues:"
    echo ""
    git --no-pager diff
    exit 1
fi
