---
name: cvdpack-difference-checker
description: Verify a pack/unpack round trip with `python -m cvdpack.checkdiff`, comparing original vs unpacked files via input/output path templates and reporting per-file diff stats (or erroring on mismatch with --error). Use to prove packing did not corrupt data.
---

# Difference checker (cvdpack.checkdiff)

## Purpose
Compare original files against their unpacked counterparts to confirm packing introduced no (or only bounded) changes. Reports `ok fraction`, mean and max abs diff per file; `--error` raises if any in-bounds pixel differs beyond `--atol`.

## When to use
- After any `pack` + `unpack` round trip, before trusting the packed dataset.
- Lossless preset: expect zero diff at default `--atol`.
- Lossy/quantized preset: expect small diffs; relax `--atol` (e.g. `0.01` for flow).

## Prerequisites
A dev checkout (`cvdpack.checkdiff` is a module). Run via `uv run -m cvdpack.checkdiff ...`.

## Commands
Run with templates (matches many files) or with concrete single-file paths.

RGB images:
```bash
uv run -m cvdpack.checkdiff \
  --input  data/TartanAir/{scene}/{split}/{vid}/image_{cam}/{frame:06d}_{cam}.{ext} \
  --output data/TartanAir_unpacked/{scene}/{split}/{vid}/image_{cam}/{frame:06d}_{cam}.{ext} \
  --error
```

Depth / seg (note the trailing `_{gt_type}` on the filename):
```bash
uv run -m cvdpack.checkdiff \
  --input  data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_{cam}_{gt_type}.{ext} \
  --output data/TartanAir_unpacked/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_{cam}_{gt_type}.{ext} \
  --error --subset gt_type=depth
```

Flow (lives in a `flow/` folder, two frame indices in the name):
```bash
uv run -m cvdpack.checkdiff \
  --input  data/TartanAir/{scene}/{split}/{vid}/flow/{frame:06d}_{f2:06d}_flow.npy \
  --output data/TartanAir_unpacked/{scene}/{split}/{vid}/flow/{frame:06d}_{f2:06d}_flow.npy \
  --error --atol 0.01
```

Single concrete files (no template / no --subset):
```bash
# single depth
uv run -m cvdpack.checkdiff \
  --input  data/TartanAir/abandonedfactory/Hard/P000/depth_left/000000_left_depth.npy \
  --output data/TartanAir_unpacked/abandonedfactory/Hard/P000/depth_left/000000_left_depth.npy

# single flow (folder is flow/, NOT flow_left/)
uv run -m cvdpack.checkdiff \
  --input  data/TartanAir/abandonedfactory/Hard/P000/flow/000000_000001_flow.npy \
  --output data/TartanAir_unpacked/abandonedfactory/Hard/P000/flow/000000_000001_flow.npy
```

## Flags
- `--input` / `--output`: path templates or concrete files.
- `--subset key=value ...`: filter matched files (e.g. `gt_type=depth`).
- `--error`: raise on any out-of-tolerance in-bounds pixel (otherwise just prints stats).
- `--atol FLOAT`: absolute tolerance (default `1e-8`; use `0.01` for lossy flow).
- `--vis all|error`: pop up matplotlib before/after/diff plots.
- `-v` / `-d`: verbosity.

## Expected output
Per file: `name name ok_pix.mean=... diffs.mean=... diffs.max=...`. With `--error`, a mismatch prints offending indices/values and raises.

## Caveats
- Only finite pixels are compared (nan/inf are masked out), which matters for the quantized preset that emits `nan` for clipped values.
- Templates must match the actual on-disk layout. Flow files are under `flow/`, not `flow_left/`.
