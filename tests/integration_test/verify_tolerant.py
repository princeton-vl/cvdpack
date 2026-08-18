import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SKIP = {"cvdpack.json"}
NUMERIC_SUFFIXES = {".png", ".jpg", ".jpeg", ".exr", ".npy", ".npz"}


def files(root: Path) -> set[Path]:
    return {
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and p.name not in SKIP
    }


def load(path: Path) -> np.ndarray | dict[str, np.ndarray]:
    match path.suffix.lower():
        case ".png" | ".jpg" | ".jpeg" | ".exr":
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"failed to read image {path}")
            return image
        case ".npy":
            return np.load(path)
        case ".npz":
            return dict(np.load(path))
        case _:
            raise ValueError(f"unsupported numeric file {path}")


def arrays_equal(blessed: np.ndarray, actual: np.ndarray, atol: float) -> str | None:
    if blessed.shape != actual.shape:
        return f"shape {blessed.shape} != {actual.shape}"
    if blessed.dtype != actual.dtype:
        return f"dtype {blessed.dtype} != {actual.dtype}"

    if blessed.dtype.kind not in "buifc":
        return None if np.array_equal(blessed, actual) else "values differ"

    finite = np.isfinite(blessed)
    if not np.array_equal(finite, np.isfinite(actual)):
        return "nan/inf in different positions"
    if not finite.all():
        if not np.array_equal(blessed[~finite], actual[~finite], equal_nan=True):
            return "nan/inf values differ"
    if not finite.any():
        return None

    errors = np.abs(
        blessed[finite].astype(np.float64) - actual[finite].astype(np.float64)
    )
    max_error = errors.max()
    if max_error <= atol:
        return None
    return f"max error {max_error} exceeds {atol}"


def difference(blessed: Path, actual: Path, atol: float) -> str | None:
    if blessed.suffix.lower() not in NUMERIC_SUFFIXES:
        return None if blessed.read_bytes() == actual.read_bytes() else "bytes differ"

    before = load(blessed)
    after = load(actual)
    if isinstance(before, dict) != isinstance(after, dict):
        return "different file structures"
    if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        return arrays_equal(before, after, atol)

    assert isinstance(before, dict)
    assert isinstance(after, dict)
    if before.keys() != after.keys():
        return "different npz keys"
    for key in sorted(before):
        error = arrays_equal(before[key], after[key], atol)
        if error is not None:
            return f"{key}: {error}"
    return None


def compare(blessed: Path, actual: Path, atol: float) -> list[str]:
    expected = files(blessed)
    produced = files(actual)
    if not expected:
        return [f"blessed tree {blessed} has no files to compare against"]
    problems = [f"missing from output: {path}" for path in sorted(expected - produced)]
    problems += [
        f"unexpected in output: {path}" for path in sorted(produced - expected)
    ]
    problems += [
        f"{path}: {error}"
        for path in sorted(expected & produced)
        if (error := difference(blessed / path, actual / path, atol)) is not None
    ]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blessed", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--atol", type=float, required=True)
    args = parser.parse_args()

    problems = compare(args.blessed, args.actual, args.atol)
    if not problems:
        print(f"all {len(files(args.blessed))} files within atol={args.atol}")
        return

    print(f"{len(problems)} files differ")
    for problem in problems[:20]:
        print(f"  {problem}")
    sys.exit(1)


if __name__ == "__main__":
    main()
