import sys
from pathlib import Path

import cv2
import numpy as np

SKIP = {"cvdpack.json"}


def pixels_equal(a: Path, b: Path) -> bool:
    x = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)
    y = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)
    if x is None or y is None:
        return False
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)


def compare(blessed: Path, actual: Path) -> list[str]:
    problems = []
    expected = {
        p.relative_to(blessed)
        for p in blessed.rglob("*")
        if p.is_file() and p.name not in SKIP
    }
    produced = {
        p.relative_to(actual)
        for p in actual.rglob("*")
        if p.is_file() and p.name not in SKIP
    }

    problems += [f"missing from output: {r}" for r in sorted(expected - produced)]
    problems += [f"unexpected in output: {r}" for r in sorted(produced - expected)]

    problems += [
        d for r in sorted(expected & produced) if (d := difference(blessed, actual, r))
    ]
    return problems


def difference(blessed: Path, actual: Path, rel: Path) -> str | None:
    a, b = blessed / rel, actual / rel
    if a.suffix.lower() == ".png":
        return None if pixels_equal(a, b) else f"pixels differ: {rel}"
    return None if a.read_bytes() == b.read_bytes() else f"bytes differ: {rel}"


def main() -> None:
    blessed, actual = Path(sys.argv[1]), Path(sys.argv[2])
    problems = compare(blessed, actual)

    n = sum(1 for p in blessed.rglob("*") if p.is_file() and p.name not in SKIP)
    if len(problems) == 0:
        print(f"all {n} files identical")
        return

    print(f"{len(problems)} of {n} files differ")
    for p in problems[:20]:
        print(f"  {p}")
    sys.exit(1)


if __name__ == "__main__":
    main()
