#!/bin/bash
set -eo pipefail

archive_run() {
    [[ "$T3_OUT" = /* && -d "$T3_OUT" ]] || return 2
    [[ ! -e "${T3_OUT}.tar.gz" && ! -e "${T3_OUT}.tar.gz.partial" ]] || return 3
    tar --exclude=checkpoint -czf "${T3_OUT}.tar.gz.partial" \
        -C "$(dirname "$T3_OUT")" "$(basename "$T3_OUT")" || return 4
    mv -n "${T3_OUT}.tar.gz.partial" "${T3_OUT}.tar.gz" || return 5
    echo "REMOTE_ARCHIVE_READY ${T3_OUT}.tar.gz"
}

if [[ "${1:-}" = --selftest ]]; then
    T3_OUT="$(mktemp -d)/synthetic_run"
    mkdir -p "$T3_OUT/checkpoint"
    touch "$T3_OUT/fixture" "$T3_OUT/checkpoint/excluded"
    archive_run
    listing=$(tar tzf "${T3_OUT}.tar.gz")
    [[ "$listing" = *synthetic_run/fixture* && "$listing" != *checkpoint* ]]
    if archive_run; then echo 'BAD archive overwrite accepted'; exit 1; fi
    T3_OUT="${T3_OUT}/missing"
    if archive_run; then echo 'BAD missing directory accepted'; exit 1; fi
    echo 'REMOTE_ARCHIVE_SELFTEST_OK good=1 bad_rejected=2'
    exit 0
fi

: "${T3_OUT:?Set a new absolute output directory}"
[[ "$T3_OUT" = /* && ! -e "$T3_OUT" && ! -e "${T3_OUT}.tar.gz" && ! -e "${T3_OUT}.tar.gz.partial" ]] || exit 3
finish() {
    train_status=$?
    trap - EXIT
    set +e
    archive_run
    archive_status=$?
    echo "T3_EXIT train_status=$train_status archive_status=$archive_status"
    if [[ $train_status != 0 ]]; then exit "$train_status"; fi
    exit "$archive_status"
}
trap finish EXIT
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
python3 /root/ops/96_cpu_quota_probe.py
python3 /root/ops/99_repeat_phase.py --execute --out "$T3_OUT"
