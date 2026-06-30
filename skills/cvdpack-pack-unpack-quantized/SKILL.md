---
name: cvdpack-pack-unpack-quantized
description: Pack and unpack a TartanAir dataset with the lossy quantized preset (tartanair_quantized.json) using libx265 / h265 video and uint16 quantization for maximum compression (~84% savings). Use when storage savings matter more than exact ground-truth fidelity.
---

# Pack / unpack (lossy, quantized, libx265)

## Purpose
Maximum compression of an RGB/Depth/Flow/Seg dataset using quantization to uint16 plus h265 (libx265) video encoding. Filesizes drop ~84% (8.6GB -> 1.3GB for one TartanAir scene).

## When to use
- You want the smallest possible packed dataset and can tolerate small, known, bounded changes to the ground truth and RGB.
- NOT for results that require bit-exact ground truth; use the lossless `cvdpack-pack-unpack-lossless` skill instead.

## Prerequisites
ffmpeg on PATH (see `cvdpack-install-setup`). For libx265 you may need `sudo apt install libx265-dev`.

## Commands
Single scene + video shown; remove `--subset` to do the whole dataset.
```bash
CVDPACK_MINOR_VIDEO_ERROR=1 uvx cvdpack pack \
  --input data/TartanAir/ --output data/TartanAir_packed/ \
  --config presets/tartanair_quantized.json \
  --tmp_folder data/tmp/ --n_workers 10 \
  --subset scene=abandonedfactory vid=P000 -v

uvx cvdpack unpack \
  --input data/TartanAir_packed --output data/TartanAir_unpacked \
  --n_workers 10 --tmp_folder data/tmp/ \
  --subset scene=abandonedfactory vid=P000 -v
```

## Expected runtime / output
- One scene: ~54s to pack, ~46s to unpack with 10 workers (AMD EPYC 7713P).
- Sizes: ~8.6GB raw -> ~1.3GB packed (~84% savings).
- A `cvdpack.json` metadata file is written into the output folder.

## Caveats / known losses
- `CVDPACK_MINOR_VIDEO_ERROR=1` is the env var that allows libx265 with yuv444p; it makes a small fraction of RGB pixel values shift by +/-1 or +/-2. (Do not use `CVDPACK_MINOR_VIDEO_ERROR_CODECS`; that name is a stale no-op.)
- The preset clips ground truth to configured min/max; out-of-bounds values come back as `nan` on unpack.
- Intermediate data is stored as uint16: flow ~0.01px precision; depth precision varies (large error past 500m).
- Industry users may need a libx265 license to unpack. libx265 is slow to encode (fast to decode).
- `--tmp_folder` is OPTIONAL (defaults to a system temp dir) but recommended for large jobs.
- Tune dynamic range vs precision, and per-channel uint16/float16/float32 choices, in the JSON config.
- Verify a round-trip with the `cvdpack-difference-checker` skill (use `--atol 0.01` for the lossy flow check).
