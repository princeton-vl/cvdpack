---
name: cvdpack-integration-test
description: Run the end-to-end pack/unpack regression test (integration_test.sh) that packs and unpacks one TartanAir scene with both the lossless and lossy presets, records sizes/timings, and asserts no unintended file changes via checkdiff. Use to validate a dev checkout.
---

# Integration test (integration_test.sh)

## Purpose
End-to-end regression check that pack + unpack works for TartanAir with no unintended file changes, for both the lossless and lossy presets.

## When to use
- Before relying on a dev checkout or before a release.
- To reproduce the timing/size numbers reported in the README.

## Prerequisites
- A dev checkout (`integration_test.sh` lives at repo root; uses `uv run`).
- ffmpeg on PATH (see `cvdpack-install-setup`).
- The TartanAir scene under `data/TartanAir/abandonedfactory/Hard/P000/`.
- GNU `time` at `/usr/bin/time` and `du` supporting `--max-depth` (GNU coreutils).

## Command
```bash
bash integration_test.sh
```

## What it does
1. Cleans `data/integration_test_tmp*/`.
2. Lossless: packs with `tartanair_floatingpoint.json`, unpacks, records timings/sizes, runs `cvdpack.checkdiff --error` on RGB, depth/seg, and flow.
3. Lossy: packs with `CVDPACK_MINOR_VIDEO_ERROR=1` + `tartanair_quantized.json`, unpacks, records sizes, runs `checkdiff` (flow with `--atol 0.01`).
4. `set -e` aborts on the first failing diff.

## Expected output
Timing files (`data/time_*_*.txt`), size files (`data/size_*.txt`), and checkdiff lines; nonzero exit if any check fails.

## Caveats
- Uses `uv run`, so run it from a dev checkout, not `uvx`.
- The script as committed uses `CVDPACK_MINOR_VIDEO_ERROR` (correct) — older docs referencing `CVDPACK_MINOR_VIDEO_ERROR_CODECS` are stale no-ops.
- Requires real TartanAir data present locally.
