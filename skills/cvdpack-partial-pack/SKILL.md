---
name: cvdpack-partial-pack
description: Run only part of the pack/unpack pipeline using --steps (quantize, pack_video, unpack_video, unquantize), e.g. just npy->png, png->mkv, mkv->png, or png->npy. Use to run stages in sequence, inspect intermediates, or skip ffmpeg.
---

# Partial pack / unpack (--steps)

## Purpose
Run individual stages of the pipeline instead of the whole thing. The four stages:
- `quantize`: npy -> png (quantized intermediate)
- `pack_video`: png -> mkv
- `unpack_video`: mkv -> png
- `unquantize`: png -> npy

## When to use
- You want to inspect or cache an intermediate representation.
- You want to skip ffmpeg entirely (only `quantize`/`unquantize` steps avoid the ffmpeg check).
- You want to run the steps on different machines or at different times.

## Prerequisites
ffmpeg on PATH only if you include `pack_video` or `unpack_video` (see `cvdpack-install-setup`).

## Commands (run in sequence)
```bash
uvx cvdpack pack   --input data/TartanAir/             --output data/TartanAir_partialpack/   --steps quantize     --n_workers 10 --cpus_per_worker 4 --config presets/tartanair_quantized.json
uvx cvdpack pack   --input data/TartanAir_partialpack/ --output data/TartanAir_packed/        --steps pack_video   --n_workers 10 --cpus_per_worker 4
uvx cvdpack unpack --input data/TartanAir_packed/      --output data/TartanAir_partialunpack/ --steps unpack_video --n_workers 10 --cpus_per_worker 4
uvx cvdpack unpack --input data/TartanAir_partialunpack/ --output data/TartanAir_unpacked/    --steps unquantize   --n_workers 10 --cpus_per_worker 4
```

## Expected runtime / output
For one scene (abandonedfactory/Hard/P000), the four steps take ~34s, ~58s, ~12s, ~23s respectively (AMD EPYC 7713P, 64-core).

## Caveats
- `--steps` accepts only `quantize`, `pack_video`, `unpack_video`, `unquantize`.
- A `--config` is required when the input folder does not already contain a `cvdpack.json`; once one stage writes metadata, later stages can read it from the input.
- `--cpus_per_worker` controls slurm CPUs and ffmpeg threads.
- `--tmp_folder` is OPTIONAL (defaults to a system temp dir).
