#!/usr/bin/env bash
# Usage: run.sh <data_dir> <out_dir> [case_name ...]
set -uo pipefail

DATA_DIR="${1:-}"
OUT_DIR="${2:-}"
shift 2 2>/dev/null || true
WANTED=("$@")

if [ -z "$DATA_DIR" ] || [ -z "$OUT_DIR" ]; then
    echo "usage: run.sh <data_dir> <out_dir> [case_name ...]" >&2
    exit 2
fi

if [ ! -f "$DATA_DIR/cases.tsv" ]; then
    echo "no case list at $DATA_DIR/cases.tsv -- is INTEGRATION_DATA staged here?" >&2
    exit 2
fi

command -v ffmpeg >/dev/null || { echo "ffmpeg not on PATH" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N_WORKERS="${N_WORKERS:-4}"
mkdir -p "$OUT_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
CASES="$DATA_DIR/cases.tsv"
cd "$REPO_ROOT"

wanted_case() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do
        [ "$w" = "$1" ] && return 0
    done
    return 1
}

resolve_config() {
    case "$1" in
        -|"") echo "" ;;
        presets/*) echo "$REPO_ROOT/cvdpack/$1" ;;
        /*) echo "$1" ;;
        *) echo "$DATA_DIR/$1" ;;
    esac
}

do_unpack() {
    local src="$1" dest="$2" config="$3" work="$4"
    local args=(--input "$src" --output "$dest" --tmp_folder "$work"
        --n_workers "$N_WORKERS" --missing_gt silent)
    [ -n "$config" ] && args+=(--config "$config")
    uv run cvdpack unpack "${args[@]}"
}

do_pack() {
    local src="$1" dest="$2" config="$3" work="$4" pack_env="$5"
    env $pack_env uv run cvdpack pack --input "$src" --output "$dest" \
        --config "$config" --tmp_folder "$work" --n_workers "$N_WORKERS" \
        --missing_gt silent
}

# --missing_gt silent lets pack exit 0 having written nothing at all.
produced_output() {
    local label="$1" dest="$2" n

    n=$(find -L "$dest" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ ! -f "$dest/cvdpack.json" ] || [ "$n" -le 1 ]; then
        echo "  FAILED: $label wrote no data to $dest ($n files, no cvdpack.json?)"
        return 1
    fi
    return 0
}

verify_tolerant() {
    local name="$1" src="$2" unpacked="$3" atol="$4"

    uv run python tests/integration_test/verify_tolerant.py \
        "$src" "$unpacked" --atol "$atol" >"$OUT_DIR/$name.diff.log" 2>&1 && {
        tail -1 "$OUT_DIR/$name.diff.log" | sed 's/^/  /'
        return 0
    }

    echo "  FAILED: differs from blessed data:"
    head -20 "$OUT_DIR/$name.diff.log" | sed 's/^/    /'
    return 1
}

verify_exact() {
    local name="$1" reference="$2" unpacked="$3" n_ref

    n_ref=$(find -L "$reference" -type f -not -name cvdpack.json | wc -l | tr -d ' ')
    if [ "$n_ref" -eq 0 ]; then
        echo "  FAILED: blessed reference $reference has no files"
        return 1
    fi

    diff -r -q -x cvdpack.json "$reference" "$unpacked" \
        >"$OUT_DIR/$name.diff.log" 2>&1 && {
        echo "  all $n_ref files identical"
        return 0
    }

    echo "  FAILED: differs from blessed reference:"
    head -20 "$OUT_DIR/$name.diff.log" | sed 's/^/    /'
    return 1
}

run_case() {
    local name="$1" blessed="$2" packed="$3" config="$4" atol="$5" pack_env="$6"
    local src="$DATA_DIR/$blessed"
    local stored="$DATA_DIR/$packed"
    local work="$OUT_DIR/$name"

    echo "=== $name ==="

    if [ ! -d "$src" ]; then
        echo "  MISSING blessed unpacked data $src"
        return 1
    fi
    if [ ! -d "$stored" ]; then
        echo "  MISSING stored packed data $stored"
        return 1
    fi
    if [ -z "$config" ]; then
        echo "  MISSING config for packing"
        return 1
    fi

    rm -rf "$work"
    mkdir -p "$work/tmp_pack" "$work/tmp_roundtrip" "$work/tmp_stored"

    echo "  pack -> unpack against blessed data"
    do_pack "$src" "$work/generated_packed" "$config" "$work/tmp_pack" "$pack_env" || return 1
    produced_output pack "$work/generated_packed" || return 1
    do_unpack "$work/generated_packed" "$work/roundtrip_unpacked" "$config" "$work/tmp_roundtrip" || return 1
    verify_tolerant "$name.roundtrip" "$src" "$work/roundtrip_unpacked" "$atol" || return 1

    echo "  stored packed -> byte-exact blessed data"
    do_unpack "$stored" "$work/stored_unpacked" "$config" "$work/tmp_stored" || return 1
    verify_exact "$name.stored" "$src" "$work/stored_unpacked"
}

passed=()
failed=()

while IFS=$'\t' read -r name blessed packed config atol pack_env; do
    case "$name" in ""|\#*) continue ;; esac
    wanted_case "$name" || continue

    config="$(resolve_config "${config:--}")"
    [ "${pack_env:--}" = "-" ] && pack_env=""

    if run_case "$name" "$blessed" "$packed" "$config" "${atol:-1e-8}" "$pack_env"; then
        passed+=("$name")
    else
        failed+=("$name")
    fi
done <"$CASES"

echo
echo "=== integration test summary ==="
echo "passed: ${#passed[@]} ${passed[*]:-}"
echo "failed: ${#failed[@]} ${failed[*]:-}"

if [ ${#passed[@]} -eq 0 ] && [ ${#failed[@]} -eq 0 ]; then
    echo "no cases ran -- $CASES selected nothing" >&2
    exit 2
fi

[ ${#failed[@]} -eq 0 ]
