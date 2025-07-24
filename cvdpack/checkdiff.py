import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from . import util

logger = logging.getLogger("cvdpack")


def vis(before, after):
    if before.ndim == 3 and before.shape[2] == 2:
        before = np.pad(
            before, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0
        )
    if after.ndim == 3 and after.shape[2] == 2:
        after = np.pad(
            after, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0
        )

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
    plt.scatter(before.flatten(), after.flatten())
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subset", type=str, nargs="*", default=None)
    parser.add_argument("--vis", type=str, default=None, choices=["all", "error"])
    parser.add_argument("--error", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-8)
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
        before = util.load_any_image(before_path)

        after_path = util.format_template(args.output, info)
        if not after_path.exists():
            raise FileNotFoundError(
                f"Got {before_path=} but {after_path=} does not exist"
            )

        after = util.load_any_image(after_path)

        if before.shape != after.shape:
            raise ValueError(
                f"Got {before.shape=} and {after.shape=} for {before_path=} and {after_path=}"
            )

        valid_mask = np.isfinite(before) & np.isfinite(after)
        ok_pix = np.isclose(before, after, atol=args.atol)

        err = not ok_pix[valid_mask].all()
        do_vis = args.vis == "all" or (args.vis == "error" and err)
        if do_vis:
            vis(before, after)

        diffs = np.abs(before[valid_mask] - after[valid_mask])
        msg = (
            f"{before_path.name} {after_path.name} {ok_pix[valid_mask].mean()=:.4f} "
            f"{diffs.mean()=:.4f} {diffs.max()=:.4f}"
        )

        if args.error and err:
            idxs = np.nonzero(~ok_pix.flatten())[:10]
            print(f"{idxs=}")
            print(f"{before.flatten()[idxs]=}")
            print(f"{after.flatten()[idxs]=}")
            raise ValueError(msg)
        else:
            print(msg)


if __name__ == "__main__":
    main()
