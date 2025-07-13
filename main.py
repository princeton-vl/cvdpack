import argparse
from typing import Literal
from enum import Enum
from pathlib import Path
import numpy as np
from string import Formatter
import re
import logging

import imageio
import cv2

logger = logging.getLogger(__name__)

PARALELL_LEVELS = [
    "scene",
    "vid",
    "cam",
    "gttype",
]

DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}

FILETYPES = [
    "png",
    "jpg",
    "npy",
    "mkv",
]

class GtType(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    FLOW = "flow"
    SURFACE_NORMAL = "surface_normal"
    SEGMENTATION = "segmentation"
    BINARY_MASK = "binary_mask"

def load_any_image(path, allow_pickle=False):
    match path.suffix:
        case ".png":
            return imageio.imread(path)
        case ".exr":
            raise NotImplementedError(f"Unhandled {path.suffix=} for {path=}")
        case ".npy":
            return np.load(path, allow_pickle=allow_pickle)
        case ".npz":
            return dict(np.load(path, allow_pickle=allow_pickle))
        case _:
            raise ValueError(f"Unhandled {path.suffix=} for {path=}")

def save_any_image(
    img: np.ndarray,
    path: Path,
):

    if img.ndim == 3 and img.shape[-1] != 3:
        raise ValueError(f"Unhandled {img.shape=} for {path=}, expected no channels (WxH) or 3 channels (WxHx3)")

    match path.suffix, img.dtype:
        case ".png", np.uint8:
            imageio.imwrite(path, img)
        case ".png", np.uint16:
            cv2.imwrite(str(path), img)
        case ".npy", _:
            np.save(path, img)
        case _:
            raise ValueError(f"Unhandled {path.suffix=} {img.dtype=}")

def quantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: str,
    quantize_method: str,
    min_orig_val: float,
    max_orig_val: float,
    out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
):
    img = load_any_image(input_img_path)

    from_dtype = img.dtype
    assert np.issubdtype(from_dtype, np.floating), f"{input_img_path=} had {from_dtype=}"

    to_dtype = DTYPE_MAP[to_dtype]
    assert np.issubdtype(to_dtype, np.integer)
    assert not np.issubdtype(to_dtype, np.signedinteger), f"Cannot quantize to signed integer: {to_dtype=}"

    # convert to 0,1 floating

    oob_mask = np.logical_or(img < min_orig_val, img > max_orig_val)
    if oob_mask.any():
        oob_pct = 100 * oob_mask.astype(np.float32).mean()
        msg = (
            f"{input_img_path=} had {img.min()=:.2f}, {img.max()=:.2f} "
            f"which exceeds quantize range [{min_orig_val:.2f}, {max_orig_val:.2f}]. {oob_pct:.2f}% were out of bounds."
        )
        match out_of_bounds_method:
            case "nan":
                img[oob_mask] = np.nan
            case "nan_warn":
                img[oob_mask] = np.nan
                logger.warning(msg + ", will be interpreted as nan")
            case "error":
                raise ValueError(msg)

    intmax = np.iinfo(to_dtype).max # the exact maxval of integer will always be used to represent "nan" or "outofbounds"
    img_quant = np.zeros_like(img, dtype=to_dtype)
    img_quant[oob_mask] = intmax

    validmax = intmax - 1
    match quantize_method:
        case "linear":
            img_quant[~oob_mask] = (
                ((img[~oob_mask] - min_orig_val) / (max_orig_val - min_orig_val)) * validmax
            ).astype(to_dtype)
        case "inv":
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_quant[~oob_mask] = ((1 / img[~oob_mask] - min_norm) / (max_norm - min_norm) * validmax).astype(to_dtype)
        case _:
            raise ValueError(f"Invalid {quantize_method=}")\

    if img_quant.ndim == 3 and img_quant.shape[-1] == 2:
        # add an empty third channel
        img_quant = np.concatenate([img_quant, np.zeros_like(img_quant[:, :, :1])], axis=2)

    save_any_image(img_quant, output_img_path)

def videopack_frameset(
    input_video_path: Path,
    output_video_path: Path,
    to_filetype: str,
):
    pass

def match_template_paths(folder: Path, template: str):
    fmt = Formatter()
    parts = []
    for lit, field, *_ in fmt.parse(template):
        parts.append(re.escape(lit))
        if field:
            parts.append(fr"(?P<{field}>\d+)")
    
    regex = "^" + "".join(parts) + "$"
    regex = re.compile(regex)
    
    def match_to_dict(m: re.Match):
        return {
            k: int(v) if v.isdigit() else v 
            for k, v in m.groupdict().items()
        }

    return [
        (match_to_dict(m), p)
        for p in folder.glob("*")
        if (m := regex.match(p.name))
    ]

def pack_frameset(
    input_path: Path,
    output_folder: Path,
    to_dtype: str,
    to_filetype: str,
    quantize_method: str,
    min_orig_val: float,
    max_orig_val: float,
):

    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)
    
    to_filetype_frame = to_filetype
    if to_filetype in ["mkv", "mp4"]:
        to_filetype_frame = "png"


    if input_path.is_dir():
        all_files = list(input_path.glob("*"))
        template = None
    elif "{" in input_path.name:
        template = input_path.name
        all_files = match_template_paths(input_path.parent, template)
    else:
        raise ValueError(f"Invalid input folder: {input_path}")

    print(len(all_files), template)

    if quantize_method is not None:
        for frame_info, frame_input_path in all_files:
            if template is not None:
                output_path = output_folder / f"{template.format(**frame_info)}"
                output_path = output_path.with_suffix("." + to_filetype_frame)
            else:
                name = frame_input_path.with_suffix("." + to_filetype_frame).name
                output_path = output_folder / name

            quantize_frame(
                frame_input_path, 
                output_path, 
                to_dtype, 
                quantize_method=quantize_method, 
                min_orig_val=min_orig_val, 
                max_orig_val=max_orig_val,
            )

    if to_filetype in ["mkv", "mp4"]:
        videopack_frameset(output_folder, output_folder, to_filetype)

def validate_args(args: argparse.Namespace):

    if args.action == "check" and (
        args.slurm_parallel is not None or args.process_parallel is not None
    ):
        raise ValueError("Cannot use --slurm_parallel or --process_parallel with --check")
    
    if not args.output.is_dir():
        args.output.mkdir(parents=True)

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["pack", "unpack", "check"])
    parser.add_argument("level", type=str, choices=["dataset", "scene", "frames"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    
    # dataset level configs - valid only for level="dataset"
    parser.add_argument("--structure", type=str)
    parser.add_argument("--config", type=str)

    # frames level configs - valid only for level="frames"
    parser.add_argument("--to_dtype", type=str, choices=DTYPE_MAP.keys())
    parser.add_argument("--to_filetype", type=str, choices=FILETYPES)
    parser.add_argument("--quantize_method", type=str, choices=["linear", "inv"])
    parser.add_argument("--min_orig_val", type=float)
    parser.add_argument("--max_orig_val", type=float)
    
    # scene level configs - valid only for level="scene"
    
    # TODO --out_of_bounds clamp clamp_warn error
    # TODO --dryrun
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallelize using a slurm array. Requires cvdpack[slurm] optional dependencies")
    parser.add_argument("--n_processes", type=int, default=None, help="Parallelize using multiprocessing")

    return validate_args(parser.parse_args())


def main():

    args = parse_args()

    match args.action, args.level:
        case "check":
            print("Checking...")
        case "pack", "dataset":
            print("Packing dataset...")
        case "pack", "scene":
            print("Packing scene...")
        case "pack", "frames":
            pack_frameset(args.input, args.output, args.to_dtype, args.to_filetype, args.quantize_method, args.min_orig_val, args.max_orig_val)
        case "unpack", "dataset":
            print("Unpacking dataset...")
        case "unpack", "scene":
            print("Unpacking scene...")

if __name__ == "__main__":
    main()