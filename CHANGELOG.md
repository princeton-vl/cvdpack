# Changelog

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
