# Changelog

## v0.7.0

### New Features

- `copy` and `unpack` accept Hugging Face dataset URLs, revisions, and subpaths with Hub support installed by default, downloading only files selected by `--subset`
- Hugging Face unpacking supports explicit upfront or per-job staging, validates SLURM scratch visibility, and removes staged data after success or failure
- Hugging Face inputs validate configs and empty subset matches before bulk transfer, retain the effective config, and record the source URL in unpacked metadata

### Fixed wrong results

- Local pack and unpack apply OR semantics to comma-separated subset values, including values containing path separators
- Local subset filters match integer template fields without conflating zero-padded strings

### Other

- `copy` no longer requires FFmpeg when it performs no video processing

## v0.6.0

### Breaking changes

- Removed the `--overwrite` flag (was accepted but had no effect)
- `copy` no longer reads a folder `cvdpack.json`, and passing `--config` with `copy` raises an error (was silently ignored)
- Only `.tar` and `.tar.gz` names are dispatched to tarball packing/unpacking

### Fixed wrong results

- A file matching multiple path templates is claimed by the most specific one (was an error)
- Subset values are substituted into competing path templates before ranking them
- Folder prefixes are flattened when unpacking a tarball

### Errors instead of silent misbehavior

- Subset keys absent from every path template are rejected
- A pack or unpack that matches no files at all raises an error
- A tarball whose members collapse to one filename is refused, and archives that would drop frames fail an audit

### Other

- CI runs lint and unit tests, a lowest-supported dependency versions check, and a staged-data integration test on every pull request
- Declared dependency version floors and a `test` extra
- Internal refactors to process dispatch and job launch, covered by pipeline-equivalence tests

## v0.5.2

- Public baseline release
