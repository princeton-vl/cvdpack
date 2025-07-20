
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from . import util

logger = logging.getLogger("cvdpack")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subset", type=str, nargs="*", default=None)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(args.loglevel)

    subset = util.parse_dictlist_strings(args.subset)
    inps = list(util.match_template_paths(args.input))

    logger.info(f"Checking {len(inps)} files")

    if subset:
        n_inps = len(inps)
        inps = [
            (info, before_path)
            for info, before_path in inps
            if util.included_in_filter(info, subset)
        ]
        n_filtered = len(inps) - n_inps
        logger.info(f"Skipped {n_filtered} files due to {subset=}")

    for info, before_path in inps:

        before = np.load(before_path)

        after_path = util.format_template(args.output, info)
        if not after_path.exists():
            raise FileNotFoundError(f"Got {before_path=} but {after_path=} does not exist")

        after = np.load(after_path)

        if before.shape != after.shape:
            raise ValueError(f"Got {before.shape=} and {after.shape=} for {before_path=} and {after_path=}")

        if not args.vis:
            close = np.isclose(before, after, atol=1e-3)
            print(f"Got {close.mean()=} for {before_path.name} {after_path.name}")
            continue

        plt.subplot(1, 4, 1)
        plt.imshow(before)
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.imshow(after)
        plt.colorbar()
        plt.subplot(1, 4, 3)
        cmap = plt.get_cmap("bwr")
        plt.imshow((before - after).clip(-1, 1), cmap=cmap)
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.imshow(np.isclose(before, after, atol=1e-3).astype(np.float32))
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.scatter(before.flatten().clip(0, 100), after.flatten().clip(0, 100))
        plt.show()

if __name__ == "__main__":
    main()