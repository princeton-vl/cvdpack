#!/usr/bin/env bash
# Usage: run.sh <data_dir> <out_dir> [case_name ...]  -- see README.md
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

verify_tolerant() {
    local name="$1" src="$2" unpacked="$3" config="$4" atol="$5"
    local checked=0 failures=0

    while IFS=$'\t' read -r gt_type template n_src n_out; do
        [ -z "$gt_type" ] && continue
        [ "$n_src" -eq 0 ] && continue

        if [ "$n_out" -ne "$n_src" ]; then
            echo "  FAILED $gt_type: input had $n_src files but unpacked $n_out"
            failures=$((failures + 1))
            continue
        fi

        echo "  checkdiff $gt_type ($n_src files)"
        checked=$((checked + 1))
        uv run -m cvdpack.checkdiff --error --atol "$atol" \
            --input "$src/$template" --output "$unpacked/$template" \
            >"$OUT_DIR/$name.$gt_type.checkdiff.log" 2>&1 && continue

        echo "  FAILED checkdiff $gt_type, tail of log:"
        tail -15 "$OUT_DIR/$name.$gt_type.checkdiff.log" | sed 's/^/    /'
        failures=$((failures + 1))
    done < <(uv run python tests/integration_test/match_counts.py "$config" "$src" "$unpacked")

    if [ "$checked" -eq 0 ]; then
        echo "  FAILED: no data_type had any files, so nothing was compared"
        return 1
    fi
    echo "  verified $checked data_type(s)"
    [ "$failures" -eq 0 ]
}

verify_exact() {
    local name="$1" reference="$2" unpacked="$3"
    local n_ref n_out

    n_ref=$(find -L "$reference" -type f -not -name cvdpack.json | wc -l | tr -d ' ')
    n_out=$(find -L "$unpacked" -type f -not -name cvdpack.json | wc -l | tr -d ' ')

    if [ "$n_ref" -eq 0 ]; then
        echo "  FAILED: blessed reference $reference has no files"
        return 1
    fi

    echo "  comparing $n_out unpacked against $n_ref blessed files"
    uv run python tests/integration_test/verify_exact.py "$reference" "$unpacked" \
        >"$OUT_DIR/$name.diff.log" 2>&1 && {
        tail -1 "$OUT_DIR/$name.diff.log" | sed 's/^/  /'
        return 0
    }

    echo "  FAILED: differs from blessed reference:"
    head -20 "$OUT_DIR/$name.diff.log" | sed 's/^/    /'
    return 1
}

run_roundtrip() {
    local name="$1" src="$2" config="$3" atol="$4" pack_env="$5" work="$6"

    env $pack_env uv run cvdpack pack --input "$src" --output "$work/packed" \
        --config "$config" --tmp_folder "$work/tmp_pack" \
        --n_workers "$N_WORKERS" --missing_gt silent || return 1

    do_unpack "$work/packed" "$work/unpacked" "" "$work/tmp_unpack" || return 1
    verify_tolerant "$name" "$src" "$work/unpacked" "$config" "$atol" || return 1

    echo "  ok: $(du -sm "$src" | cut -f1)MB -> $(du -sm "$work/packed" | cut -f1)MB packed"
}

run_case() {
    local name="$1" mode="$2" input="$3" reference="$4" config="$5" atol="$6" pack_env="$7"
    local src="$DATA_DIR/$input"
    local work="$OUT_DIR/$name"

    echo "=== $name ($mode): $input ==="

    if [ ! -d "$src" ]; then
        echo "  MISSING input $src"
        return 1
    fi

    rm -rf "$work"
    mkdir -p "$work/tmp_pack" "$work/tmp_unpack"

    case "$mode" in
        roundtrip)
            run_roundtrip "$name" "$src" "$config" "$atol" "$pack_env" "$work"
            ;;
        unpack)
            if [ ! -d "$DATA_DIR/$reference" ]; then
                echo "  MISSING blessed reference $DATA_DIR/$reference"
                return 1
            fi
            do_unpack "$src" "$work/unpacked" "$config" "$work/tmp_unpack" || return 1
            verify_exact "$name" "$DATA_DIR/$reference" "$work/unpacked"
            ;;
        *)
            echo "  unknown mode '$mode', expected roundtrip or unpack"
            return 1
            ;;
    esac
}

passed=()
failed=()

while IFS=$'\t' read -r name mode input reference config atol pack_env; do
    case "$name" in ""|\#*) continue ;; esac
    wanted_case "$name" || continue

    config="$(resolve_config "${config:--}")"
    [ "${pack_env:--}" = "-" ] && pack_env=""

    if run_case "$name" "$mode" "$input" "${reference:--}" "$config" "${atol:-1e-8}" "$pack_env"; then
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
