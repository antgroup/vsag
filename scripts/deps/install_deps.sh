#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

run_privileged_script() {
    local target_script="$1"

    if [[ $(id -u) -eq 0 ]]; then
        bash "${target_script}"
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        sudo bash "${target_script}"
        return
    fi

    echo "Root or sudo is required to install Linux dependencies." >&2
    exit 1
}

case "$(uname -s)" in
    Darwin)
        bash "${script_dir}/install_deps_macos.sh"
        ;;
    Linux)
        distro_id=""
        distro_like=""
        if [[ -f /etc/os-release ]]; then
            # shellcheck disable=SC1091
            . /etc/os-release
            distro_id="${ID:-}"
            distro_like="${ID_LIKE:-}"
        fi

        distro_tokens=" ${distro_id} ${distro_like} "
        if [[ "${distro_tokens}" == *" ubuntu "* || "${distro_tokens}" == *" debian "* ]]; then
            run_privileged_script "${script_dir}/install_deps_ubuntu.sh"
        elif [[ "${distro_tokens}" == *" centos "* || "${distro_tokens}" == *" rhel "* || "${distro_tokens}" == *" rocky "* || "${distro_tokens}" == *" alinux "* || "${distro_tokens}" == *" anolis "* ]]; then
            run_privileged_script "${script_dir}/install_deps_centos.sh"
        else
            echo "Unsupported Linux distribution: ID='${distro_id}' ID_LIKE='${distro_like}'." >&2
            echo "Run a platform-specific script under scripts/deps/ directly." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported operating system: $(uname -s)" >&2
        exit 1
        ;;
esac
