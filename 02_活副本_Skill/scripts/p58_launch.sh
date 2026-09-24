#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
if [[ $# -ne 2 ]]; then
    echo 'Usage: bash p58_launch.sh NEW_ABSOLUTE_OUTPUT legacy-v1|warmed-v2' >&2
    exit 3
fi
exec python3 "$(dirname "${BASH_SOURCE[0]}")/58_optimizer_screen.py" --execute --out "$1" --protocol "$2"
