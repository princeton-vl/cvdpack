---
name: cvdpack-reorganize-dataset
description: Reorganize a dataset's directory/file layout with `cvdpack copy`, remapping paths via input/output templates with placeholders like {scene}/{split}/{vid}/{cam}/{gt_type}/{frame}. Use to restructure a dataset without packing or compression.
---

# Reorganize a dataset (cvdpack copy)

## Purpose
Move/rename files from one directory layout to another by matching an input path template and writing to an output path template. No compression; this is a pure copy/restructure.

## When to use
- You want to change a dataset's folder structure (e.g. group by camera instead of by gt_type) without packing.
- As a preprocessing step before packing if the source layout does not match a preset.

## Prerequisites
None (no ffmpeg needed for `copy`).

## Command
```bash
uvx cvdpack copy \
  --input  data/TartanAir/{scene}/{split}/{vid}/{gt_type}_{cam}/{frame:06d}_*.{ext} \
  --output data/TartanAir_split/{scene}/{split}_{vid}/{cam}/{gt_type}/{frame:04d}.{ext}
```

## How it works
- Placeholders in `--input` (e.g. `{scene}`, `{vid}`, `{cam}`, `{gt_type}`, `{frame:06d}`, `{ext}`) are matched against real paths; matched values are substituted into `--output`.
- The copy must be one-to-one: if two inputs map to the same output (or vice versa) it errors.
- `--frame:06d` vs `--frame:04d` lets you re-zero-pad frame numbers across the move.

## Expected output
Files copied into the new layout. No `cvdpack.json` is written for `copy`.

## Caveats
- `--n_workers` / `--parallel_mode` do NOT work with `copy` (it errors); copy is single-process.
- Some layouts that store multiple gt types in one folder (e.g. TartanAir flow + mask) can be hard to express in one template; see the `cvdpack-extract-subset` skill for filtering.
