# AGENTS.md — cvdpack agent skills

This repo ships reusable, agent-agnostic **skills** under `skills/`. Each skill is one
workflow distilled from `README.md` into a self-contained `skills/<name>/SKILL.md`.

## Format choice
Each skill is `skills/<name>/SKILL.md`: plain Markdown with minimal YAML frontmatter
(`name`, `description`). This is exactly the [Claude Code / Claude Agent Skills](https://code.claude.com/docs/en/skills)
convention (folder name == `name`; `description` says what it does and when to use it),
**and** it is just Markdown that any agent can read. This `AGENTS.md` is the
[agents.md](https://agents.md) index that OpenAI Codex and other agents.md-aware tools read:
Codex has no native skills loader, so it discovers the skills here and opens the referenced
`SKILL.md` files as plain Markdown. No bespoke schema — both ecosystems already understand
this SKILL.md + AGENTS.md pairing.

## How each agent consumes these
- **Claude Code**: auto-loads each `skills/<name>/SKILL.md` (name + description) and invokes
  by description match. (Move/symlink under `.claude/skills/` if you prefer that location.)
- **OpenAI Codex / agents.md tools**: read this `AGENTS.md`, then open the relevant
  `SKILL.md` by its path below.

## Conventions baked into every skill
- Run cvdpack with no install via `uvx cvdpack ...`; use `uv run cvdpack ...` in a dev checkout.
- The lossy-RGB env var is `CVDPACK_MINOR_VIDEO_ERROR` (0/1). `CVDPACK_MINOR_VIDEO_ERROR_CODECS` is a stale no-op — do not use it.
- `--tmp_folder` is OPTIONAL (defaults to a system temp dir); pass a clean path like `/scratch/$USER/cvdpack_tmp/` for large/slurm jobs.
- `ffmpeg` must be on PATH for any step that does `pack_video`/`unpack_video`.

## Skill index
| Skill | When to use | Path |
|-------|-------------|------|
| install-setup | First, on a fresh machine / CI: install ffmpeg + uv, run via `uvx`, optionally install the package. | `skills/install-setup/SKILL.md` |
| pack-unpack-quantized | Lossy, maximum compression (~84%) with libx265 + uint16 quantization (`tartanair_quantized.json`). | `skills/pack-unpack-quantized/SKILL.md` |
| pack-unpack-lossless | Zero intended changes (~48%) with the floating-point preset (`tartanair_floatingpoint.json`). | `skills/pack-unpack-lossless/SKILL.md` |
| reorganize-dataset | Restructure a dataset's layout with `cvdpack copy` and path templates. | `skills/reorganize-dataset/SKILL.md` |
| extract-subset | Pull out a filtered slice with `cvdpack copy --subset`. | `skills/extract-subset/SKILL.md` |
| partial-pack | Run individual pipeline stages via `--steps quantize/pack_video/unpack_video/unquantize`. | `skills/partial-pack/SKILL.md` |
| slurm-pack | Massively parallel pack/unpack on a SLURM cluster via `--parallel_mode slurm`. | `skills/slurm-pack/SKILL.md` |
| difference-checker | Verify a round trip with `python -m cvdpack.checkdiff`. | `skills/difference-checker/SKILL.md` |
| integration-test | Run the end-to-end regression test `integration_test.sh`. | `skills/integration-test/SKILL.md` |
