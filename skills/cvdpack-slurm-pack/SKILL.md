---
name: cvdpack-slurm-pack
description: Pack/unpack a large dataset massively in parallel on a SLURM cluster via --parallel_mode slurm with submitit, including lossless and lossy variants. Use for full-dataset jobs too big for a single machine.
---

# SLURM pack / unpack (massively parallel)

## Purpose
Distribute pack/unpack jobs across a SLURM cluster instead of one machine's process pool, using submitit. Works off the shelf on princeton-vl's cluster; customize paths and slurm args for your own.

## When to use
- Packing/unpacking a full dataset that is too large for one node.
- Any job where you want hundreds of workers.

## Prerequisites
- ffmpeg on PATH on the worker nodes.
- The slurm extras installed: `uv pip install cvdpack[slurm]` (provides `submitit`).
- A clean writable `--tmp_folder` on each node, e.g. `/scratch/$USER/cvdpack_tmp/`.

## Commands

### Lossless full-dataset pack
```bash
CVDPACK_MINOR_VIDEO_ERROR=0 screen uvx cvdpack pack \
  --input /n/fs/circuitnn/datasets/TartanAir \
  --output /n/fs/scratch/$USER/data/TartanAir_packed \
  --config presets/tartanair_floatingpoint.json \
  --tmp_folder /scratch/$USER/cvdpack_tmp/ \
  --parallel_mode slurm --n_workers 200 \
  --slurm_args slurm_account=allcs -v
```

### Lossy full-dataset pack
```bash
CVDPACK_MINOR_VIDEO_ERROR=1 screen uvx cvdpack pack \
  --input /n/fs/circuitnn/datasets/TartanAir \
  --output /n/fs/scratch/$USER/data/TartanAir_packed \
  --config presets/tartanair_quantized.json \
  --tmp_folder /scratch/$USER/cvdpack_tmp/ \
  --parallel_mode slurm --n_workers 200 \
  --slurm_args slurm_account=allcs -v
```

### Unpack (same for either variant)
```bash
screen uvx cvdpack unpack \
  --input /n/fs/scratch/$USER/data/TartanAir_packed \
  --output /n/fs/scratch/$USER/data/TartanAir_unpacked \
  --tmp_folder /scratch/$USER/cvdpack_tmp/ \
  --parallel_mode slurm --n_workers 200 \
  --slurm_args slurm_account=allcs
```

You can also pin specific nodes:
`--slurm_args slurm_account=pvl slurm_nodelist=node007,node[020-026],node[101-104],node403`

## Expected output
- A timestamped `*_cvdpack_pack` / `*_cvdpack_unpack` log folder under the output dir holds per-array-task `ID_log.out` / `ID_log.err`.
- A `cvdpack.json` metadata file is written into the output folder.
- If any array task crashes, the run raises and names the crashed job IDs and the log folder to inspect.

## Caveats
- Use a clean tmp path like `/scratch/$USER/cvdpack_tmp/` (no spaces). Do NOT use the broken README form `/scratch/$USER/uvx cvdpack_tmp/`.
- Use `CVDPACK_MINOR_VIDEO_ERROR` (0 = lossless, 1 = allow lossy libx265). `CVDPACK_MINOR_VIDEO_ERROR_CODECS` is a stale no-op.
- `--n_workers` above `CVDPACK_SLURM_ARRAY_MAX` (default 500) is capped; raise that env var if your cluster allows larger arrays.
- Run under `screen` so the launcher survives disconnects.
- `--slurm_args` are space-separated `key=value` pairs forwarded to submitit's `update_parameters`.
