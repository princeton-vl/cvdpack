---
name: cvdpack-pack-unpack-lossless
description: Pack and unpack a TartanAir dataset with the lossless floating-point preset (tartanair_floatingpoint.json), with zero intended changes to images or ground truth (~48% savings). Use when exact fidelity matters. This path is WIP and less space-efficient.
---

# Pack / unpack (lossless, floating point)

## Purpose
Reorganize and compress a dataset with zero intended changes to images or ground truth, using lossless FFV1 video and float-preserving packing.

## When to use
- You need a bit-exact round trip of RGB and ground truth.
- You accept modest savings (~48%) in exchange for fidelity.

## Prerequisites
ffmpeg on PATH (see `cvdpack-install-setup`).

## Commands
Single scene + video shown; remove `--subset` to do the whole dataset.
```bash
uvx cvdpack pack \
  --input data/TartanAir/ --output data/TartanAir_packed/ \
  --config presets/tartanair_floatingpoint.json \
  --tmp_folder data/tmp/ --n_workers 10 \
  --subset scene=abandonedfactory vid=P000 -v

uvx cvdpack unpack \
  --input data/TartanAir_packed --output data/TartanAir_unpacked \
  --n_workers 10 --tmp_folder data/tmp/ \
  --subset scene=abandonedfactory vid=P000 -v
```

## Expected runtime / output
- One scene (abandonedfactory/Hard/P000): ~93s to pack, ~36s to unpack (AMD EPYC 7713P).
- Sizes: ~8.6GB raw -> ~4.5GB packed (~48% savings).
- A `cvdpack.json` metadata file is written into the output folder.

## Caveats
- This setting is WIP and not very space-efficient. Float32 is currently reinterpret-cast to uint16 video, producing stripey patterns that compress poorly; video compression may add little over plain PNGs here.
- `--tmp_folder` is OPTIONAL (defaults to a system temp dir) but recommended for large jobs.
- Verify a round-trip with the `cvdpack-difference-checker` skill (default `--atol`, expect zero diff).
