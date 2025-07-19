
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from .main import match_template_paths, format_template

logger = logging.getLogger("cvdpack")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_template", type=Path)
    parser.add_argument("output_template", type=Path)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
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

    inps = list(match_template_paths(args.input_template))

    print(f"Checking {len(inps)} files")

    for info, before_path in inps:
        before = np.load(before_path)

        after_path = format_template(args.output_template, info)
        if not after_path.exists():
            raise FileNotFoundError(f"Got {before_path=} but {after_path=} does not exist")

        after = np.load(after_path)

        before = np.load(args.input_template)
        after = np.load(args.output_template)

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