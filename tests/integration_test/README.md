# Integration fixtures

`cases.tsv` lives with the staged fixture data. Its columns are:

```text
name	blessed_unpacked	stored_packed	config	atol	pack_env
```

Each case must provide a blessed unpacked tree and a stored packed tree. Both
are staged copies of real data, kept as the correct answer.

`run.sh` performs two checks for each selected case:

1. packs the blessed tree, unpacks it, and measures numeric reconstruction error against `atol`;
2. unpacks the stored packed tree and requires every resulting file to byte-match the blessed tree.

`config` is either a repository preset or a path relative to the fixture root. `pack_env` is `-` when no environment override is required.

`atol` is a pure absolute tolerance, so it must exceed one quantization step of
the lossiest packing method the config uses. Repacking already-quantized data
lands on an adjacent bin wherever a value sits on a rounding boundary.

## Staging a case

The blessed tree must be the unpack of the stored packed tree, never the raw
input, because unpack re-encodes PNGs and reformats text and so never
reproduces raw bytes. Byte-exactness also makes a blessed tree specific to the
image encoder that wrote it, so restage a case whenever its encoder changes.

```bash
uv run cvdpack pack --input <raw> --output <fixture>/packed --config <config>
uv run cvdpack unpack --input <fixture>/packed --output <fixture>/unpacked --config <config>
```

## Auditing an existing release

A cvdpack before the `pack_tarball` arcname fix named every member of a frame
tarball after the tarball itself, so unpacking keeps only one frame. Unpack now
refuses such an archive; to find them in an already-published dataset without
unpacking it, look for members that share a basename:

```bash
find <packed> -name '*.tar.gz' -exec sh -c '
    tar tzf "$1" | grep -v "/$" | sed "s|.*/||" | sort | uniq -d | grep -q . &&
        echo "$1: members collapse to one filename, repack from the original frames"' _ {} \;
```
