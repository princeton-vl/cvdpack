---
name: cvdpack-install-setup
description: Install the prerequisites for cvdpack (ffmpeg + uv), then run it via uvx with no install, or optionally install the cvdpack package for the Python API. Use this before running any other cvdpack workflow.
---

# Install / setup cvdpack

## Purpose
Get a machine ready to pack/unpack datasets with cvdpack. cvdpack shells out to `ffmpeg` for all video operations, and is published to PyPI so it can be run with no install via `uvx`.

## When to use
- First time on a machine, or in CI, before any `pack`/`unpack`/`copy`/`checkdiff` workflow.
- When a command fails with `ffmpeg is required for video operations but was not found`.

## Prerequisites
None beyond a shell and internet access.

## Steps

### 1. Install ffmpeg into your PATH
uv/pip will NOT install ffmpeg for you. Pick the one for your OS:
```bash
conda install ffmpeg
sudo apt install ffmpeg
brew install ffmpeg
```
For the lossy libx265 path you may also need the encoder libs:
```bash
sudo apt install libx265-dev
```

### 2. Install uv
Follow https://docs.astral.sh/uv/getting-started/installation/ (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 3. Run cvdpack with no install
```bash
uvx cvdpack --help
```
`uvx` fetches and runs the published package each time; no project install is needed.

### 4. (Optional) Install the package for the Python API
Only needed if you want to `import cvdpack` or use the Python interface:
```bash
uv pip install cvdpack
# or
pip install cvdpack
```

### 5. (Developer checkout only)
If you cloned the repo and want to run your local code:
```bash
git clone https://github.com/princeton-vl/cvdpack.git
cd cvdpack
uv pip install -e .[dev]
```
In a dev checkout, run every example as `uv run cvdpack ...` instead of `uvx cvdpack ...`.

## Expected output
`uvx cvdpack --help` prints the argparse usage with the `pack`/`unpack`/`copy` actions.

## Caveats
- ffmpeg must be on PATH; `uvx cvdpack pack/unpack` validates this at startup unless `--steps` avoids `pack_video`/`unpack_video`.
- slurm parallelism needs the extras: install `cvdpack[slurm]` (provides `submitit`).
- Use `uvx cvdpack` for ad-hoc use; use `uv run cvdpack` only inside a dev checkout.
