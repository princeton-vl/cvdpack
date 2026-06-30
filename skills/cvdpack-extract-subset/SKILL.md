---
name: cvdpack-extract-subset
description: Extract a filtered subset of a dataset with `cvdpack copy --subset`, selecting specific scenes/splits/vids/gt_types/cameras via key=value (comma-separated) filters. Use to pull out just part of a dataset.
---

# Extract a subset of a dataset (cvdpack copy --subset)

## Purpose
Copy out only the files matching a filter, e.g. one scene, the Hard split, a couple of videos, only image+depth, left camera.

## When to use
- You want a small slice of a large dataset for testing or sharing.
- You want to filter while reorganizing layout.

## Prerequisites
None (no ffmpeg needed for `copy`).

## Command
```bash
uvx cvdpack copy \
  --input  data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_*.{ext} \
  --output data/TartanAir_split/{} \
  --subset scene=abandonedfactory split=Hard vid=P000,P001 gt_type=image,depth cam=left
```

## How it works
- `--subset` takes space-separated `key=value` pairs; keys must match placeholders in the template. Multiple values per key are comma-separated (`vid=P000,P001`).
- `--output data/TartanAir_split/{}` is a folder shorthand: when one side is `{}`, cvdpack infers the matching template from the other side, so the source layout is preserved under the new root.

## Expected output
Only the matching files are copied. No `cvdpack.json` is written for `copy`.

## Caveats
- `--n_workers` / `--parallel_mode` do NOT work with `copy`.
- Layouts that mix gt types in one folder (e.g. TartanAir flow + mask) can be tricky to subset cleanly in a single template.
- You can also pass fully concrete paths and skip `--subset` entirely.
