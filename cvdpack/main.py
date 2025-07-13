import argparse
from typing import Literal
from enum import Enum
from pathlib import Path
import numpy as np
from string import Formatter
import re
import logging
import subprocess

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
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

VIDEO_SUFFIXES = [
    "mkv",
    "mp4",
]

FILETYPES = [
    "png",
    "jpg",
    "npy",
    "mkv",
]

ENCODER_ARGS = {
    "ffv1": "-c:v ffv1 -level 3 -g 1 -slices 4 -threads 4 -slicecrc 1",
    "libx265": "-c:v libx265 -x265-params lossless=1 -preset veryslow",
}

class GtType(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    FLOW = "flow"
    SURFACE_NORMAL = "surface_normal"
    SEGMENTATION = "segmentation"
    BINARY_MASK = "binary_mask"

    @classmethod
    def from_str(cls, s: str):
        return cls(s.lower())

class QuantizeMethod(Enum):
    LINEAR = "linear"
    SYM_SQRT = "sym_sqrt"
    INV = "inv"

    @classmethod
    def from_str(cls, s: str):
        return cls(s.lower())

PROPS_TO_ENCODER_PIXFMT = {
    ('uint8', 3): ("libx265", "yuv444p"),
    ('uint16', 1): ("ffv1", "gray16le"),
    ('uint16', 3): ("ffv1", "rgb48"),
    ('bool', 1): ("libx265", "gray"),
}

def load_any_image(path, allow_pickle=False):
    match path.suffix:
        case ".png":
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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
        case ".png", np.uint8 | np.uint16:
            cv2.imwrite(str(path), img)
        case ".npy", _:
            np.save(path, img)
        case _:
            raise ValueError(f"Unhandled {path.suffix=} {img.dtype=}")
        
    assert path.exists(), f"Failed to save {path=}"

def sym_sqrt(img: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(img)) * np.sign(img)

def normalize_vals(
    img: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    quantize_method: QuantizeMethod,
) -> np.ndarray:
    
    match quantize_method:
        case QuantizeMethod.LINEAR:
            img_norm = (img - min_orig_val) / (max_orig_val - min_orig_val)
            return img_norm
        case QuantizeMethod.SYM_SQRT:
            img_norm = sym_sqrt(img)
            sqrt_min = sym_sqrt(min_orig_val)
            sqrt_max = sym_sqrt(max_orig_val)
            img_norm = (img_norm - sqrt_min) / (sqrt_max - sqrt_min)
            return img_norm
        case QuantizeMethod.INV:
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_norm = (1 / img - min_norm) / (max_norm - min_norm)
            return img_norm
        case _:
            raise ValueError(f"Invalid {quantize_method=}")
        
def unnormalize_vals(
    img: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    quantize_method: QuantizeMethod,
) -> np.ndarray:
    
    match quantize_method:
        case QuantizeMethod.LINEAR:
            return img * (max_orig_val - min_orig_val) + min_orig_val
        case QuantizeMethod.INV:
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_inv = (img - min_norm) / (max_norm - min_norm)
            return 1 / img_inv
        case _:
            raise ValueError(f"Invalid {quantize_method=}")

def quantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: str,
    quantize_method: QuantizeMethod,
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

    img_norm = normalize_vals(img, min_orig_val, max_orig_val, quantize_method)

    intmax = np.iinfo(to_dtype).max # the exact maxval of integer will always be used to represent "nan" or "outofbounds"
    img_quant = np.zeros_like(img, dtype=to_dtype)
    img_quant[oob_mask] = intmax
    img_quant[~oob_mask] = (img_norm[~oob_mask] * (intmax - 1)).astype(to_dtype) + 1

    logger.info(f"Quantizing {input_img_path=} to {output_img_path=}, {img.min()=:.2f}, {img.max()=:.2f}, {img_quant.min()=:.2f}, {img_quant.max()=:.2f}")

    if (
        output_img_path.suffix == ".png"
        and img_quant.ndim == 3 
        and img_quant.shape[-1] == 2
    ):
        # add an empty third channel for compatibility with png formats
        img_quant = np.concatenate([img_quant, np.zeros_like(img_quant[:, :, :1])], axis=2)

    save_any_image(img_quant, output_img_path)

def _curlyframe_to_ffmpeg_frametemplate(input_path: Path, as_glob: bool = False):
    newname = re.sub(
        r"\{frame:(0\d+)d\}",  # e.g. {frame:06d}
        lambda m: "*" if as_glob else f"%{m.group(1)}d",
        input_path.name,
    )
    if "{" in newname:
        raise ValueError(f"Input frames path  must not contain templates besides {{frame}}, got {input_path=}")
    return str(input_path.parent / newname)

def unpack_video(
    input_video_path: Path,
    output_frames_path_template: Path,
    ffmpeg: str = "ffmpeg",
):
    
    output_path_ffmpeg = _curlyframe_to_ffmpeg_frametemplate(output_frames_path_template)

    command = f"{ffmpeg} -y -hide_banner -i {input_video_path} {output_path_ffmpeg}"
    logger.info(f"Unpacking {input_video_path=} to {output_frames_path_template=}, {command=}")
    subprocess.check_output(command.split())

def pack_video(
    input_frames_path: Path,
    output_video_path: Path,
    ffmpeg: str = "ffmpeg",
):
    
    matched = match_template_paths(input_frames_path.parent, input_frames_path.name)
    if len(matched) == 0:
        raise ValueError(f"No frames found in {input_frames_path=}")
    first = load_any_image(matched[0][1])
    encoder, pix_fmt = PROPS_TO_ENCODER_PIXFMT[(str(first.dtype), first.shape[-1])]
    encoder_args = ENCODER_ARGS[encoder]

    input_frames_ffmpeg = _curlyframe_to_ffmpeg_frametemplate(input_frames_path, as_glob=True)
    command = f"{ffmpeg} -y -hide_banner -pattern_type glob -i {input_frames_ffmpeg} {encoder_args} -pix_fmt {pix_fmt} -an {output_video_path}"
    logger.info(f"Packing {input_frames_path=} to {output_video_path=}, {command=}")
    subprocess.check_output(command.split())

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
        for p in sorted(list(folder.glob("*")))
        if (m := regex.match(p.name))
    ]

def pack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: str,
    quantize_method: str,
    min_orig_val: float,
    max_orig_val: float,
    out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
):

    all_files = match_template_paths(input_path_template.parent, input_path_template.name)
    logger.info(f"Found {len(all_files)} frames in {input_path_template=}")

    for frame_info, frame_input_path in all_files:

        output_path = (
            output_path_template.parent 
            / output_path_template.name.format(**frame_info)
        )

        quantize_frame(
            frame_input_path, 
            output_path, 
            to_dtype, 
            quantize_method=quantize_method, 
            min_orig_val=min_orig_val, 
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: str,
    quantize_method: str,
    min_orig_val: float,
    max_orig_val: float,
):
    
    assert input_img_path.suffix == ".png"
    img = load_any_image(input_img_path)

    assert np.issubdtype(img.dtype, np.integer), f"{input_img_path=} had {img.dtype=}"

    img_unquant = unnormalize_vals(img, min_orig_val, max_orig_val, quantize_method)
    img_unquant = img_unquant.astype(DTYPE_MAP[to_dtype])

    isnan = (img == np.iinfo(img.dtype).max)
    img_unquant[isnan] = np.nan

    min = img[~isnan].min()
    max = img[~isnan].max()
    logger.info(f"Unquantizing {input_img_path=} to {output_img_path=}, {isnan.mean()=:.2f}, {min=:.2f}, {max=:.2f}, {img_unquant.min()=:.2f}, {img_unquant.max()=:.2f}")

    save_any_image(img_unquant, output_img_path)

def unpack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: str,
    quantize_method: str,
    min_orig_val: float,
    max_orig_val: float,
):
    
    assert "{" in input_path_template.name, f"Input path must contain a template: {input_path_template=}"
    assert "{" in output_path_template.name, f"Output path must contain a template: {output_path_template=}"
    all_files = match_template_paths(input_path_template.parent, input_path_template.name)
    logger.info(f"Found {len(all_files)} frames in {input_path_template=}")

    for frame_info, frame_input_path in all_files:

        output_path = (
            output_path_template.parent 
            / output_path_template.name.format(**frame_info)
        )

        unquantize_frame(
            frame_input_path, 
            output_path, 
            to_dtype, 
            quantize_method=quantize_method, 
            min_orig_val=min_orig_val, 
            max_orig_val=max_orig_val,
        )

def validate_args(args: argparse.Namespace):

    if args.action == "check" and (
        args.slurm_parallel is not None or args.process_parallel is not None
    ):
        raise ValueError("Cannot use --slurm_parallel or --process_parallel with --check")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["pack", "unpack", "check"])
    parser.add_argument("level", type=str, choices=["dataset", "scene", "frames"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    
    # frames level configs - valid only for level="frames"
    parser.add_argument("--to_dtype", type=str, choices=DTYPE_MAP.keys(), default=None)
    parser.add_argument("--quantize_method", type=QuantizeMethod.from_str, default=None, choices=list(QuantizeMethod))
    parser.add_argument("--min_orig_val", type=float, default=None)
    parser.add_argument("--max_orig_val", type=float, default=None)
    parser.add_argument("--out_of_bounds_method", type=str, choices=["nan", "nan_warn", "error"], default="nan_warn")
    
    # scene level configs - valid only for level="scene"
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallelize using a slurm array. Requires cvdpack[slurm] optional dependencies")
    parser.add_argument("--n_processes", type=int, default=None, help="Parallelize using multiprocessing")

    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--log_level", type=str, default="INFO")

    return validate_args(parser.parse_args())


def main():

    args = parse_args()

    logging.basicConfig(level=args.log_level)


    match args.action, args.level, args.input.suffix, args.output.suffix:
        case "pack", "frames", _, ".png":
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            pack_frameset(
                args.input, 
                args.output, 
                args.to_dtype, 
                args.quantize_method, 
                args.min_orig_val, 
                args.max_orig_val,
                args.out_of_bounds_method,
            )
        case "pack", "frames", ".png", ".mkv":
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            pack_video(
                args.input, 
                args.output, 
            )
        case "unpack", "frames", ".png", _:
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            unpack_frameset(
                args.input, 
                args.output, 
                args.to_dtype, 
                args.quantize_method, 
                args.min_orig_val, 
                args.max_orig_val,
            )
        case "unpack", "frames", ".mkv", ".png":
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            unpack_video(
                args.input, 
                args.output, 
            )
        case _:
            raise ValueError(f"Invalid {args.action=} {args.level=} {args.input.suffix=} {args.output.suffix=}")

if __name__ == "__main__":
    main()