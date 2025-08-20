#!/bin/bash

function get_os() {
    os_name=$(cat /etc/*-release | grep '^NAME=' | awk -F= '{print $2}' | tr -d '"')
    os_version=$(cat /etc/*-release | grep '^VERSION_ID=' | awk -F= '{print $2}' | tr -d '"')
    echo "- OS: $os_name $os_version"
}

function get_vsag() {
    vsag_version=$(git describe --tags --always --dirty --match "v*")
    echo "- vsag version: $vsag_version"
}

function get_compiler() {
    compiler_version=$(${CXX:-c++} --version | head -n 1)
    echo "- compiler version: $compiler_version"
}

get_os
get_vsag
get_compiler
