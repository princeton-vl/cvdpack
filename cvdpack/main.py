import argparse
import copy
import json
import logging
import multiprocessing
import re
import shutil
import subprocess
from enum import Enum
import itertools
from pathlib import Path
from string import Formatter
from typing import Literal, Callable

import cv2
import numpy as np
from cvdpack import __version__
import time
from tqdm import tqdm

try:
    import submitit
except ImportError:
    submitit = None

logger = logging.getLogger("cvdpack")

SLURM_ARRAY_MAX = 500

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
    "libx265": "-c:v libx265 -x265-params lossless=1 -preset slow",
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
    SYM_SQRT = "symsqrt"
    INV = "inv"
    CHECKBOUNDS = "checkbounds"

    @classmethod
    def from_str(cls, s: str):
        return cls(s.lower())


PROPS_TO_ENCODER_PIXFMT = {
    ("uint8", 3): ("libx265", "yuv444p"),
    ("uint16", 1): ("ffv1", "gray16le"),
    ("uint16", 3): ("ffv1", "rgb48"),
    ("uint8", 1): ("libx265", "gray"),
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
        raise ValueError(
            f"Unhandled {img.shape=} for {path=}, expected no channels (WxH) or 3 channels (WxHx3)"
        )

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
            raise ValueError(f"Invalid {quantize_method=} {type(quantize_method)=}")


def img_quant_to_orig(
    img: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    from_dtype: np.dtype,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    unpack_channels_last: int | None = None,
) -> np.ndarray:
    
    assert np.issubdtype(from_dtype, np.unsignedinteger), f"{from_dtype=}"
    from_max = np.iinfo(from_dtype).max

    match quantize_method:
        case QuantizeMethod.CHECKBOUNDS:
            img = img.astype(to_dtype)
        case QuantizeMethod.LINEAR:
            img_norm = img.astype(np.float64) / from_max
            img_orig = (img_norm * (max_orig_val - min_orig_val) + min_orig_val)
            img = img_orig.astype(to_dtype)
        case QuantizeMethod.INV:
def img_orig_to_quant(
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_unmap = (img_norm * (max_norm - min_norm) + min_norm)
            img_orig = 1 / img_unmap
            img = img_orig.astype(to_dtype)
        case _:
            raise ValueError(f"Invalid {quantize_method=}")
        
    if unpack_channels_last is not None:
        # needed for cases like flow, which can be 2 channel, but will have been promoted to a 3 channel png/mkv
        assert img.ndim == 3, f"{img.ndim=}"
        assert img.shape[-1] >= unpack_channels_last, f"{img.shape=}"
        img = img[..., :unpack_channels_last]
    
    return img
def img_orig_to_quant(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
):
    img = load_any_image(input_img_path)

    from_dtype = img.dtype

    assert np.issubdtype(to_dtype, np.integer)
    assert not np.issubdtype(to_dtype, np.signedinteger), (
        f"Cannot quantize to signed integer: {to_dtype=}"
    )

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

    if quantize_method == QuantizeMethod.CHECKBOUNDS:
        assert np.issubdtype(from_dtype, np.integer), (
            f"{input_img_path=} had {from_dtype=}"
        )
        img_quant = img.astype(to_dtype)
    else:
        assert np.issubdtype(from_dtype, np.floating), (
            f"{input_img_path=} had {from_dtype=}"
        )
        assert np.issubdtype(to_dtype, np.integer), f"{input_img_path=} had {to_dtype=}"
        img_norm = normalize_vals(img, min_orig_val, max_orig_val, quantize_method)
        intmax = np.iinfo(
            to_dtype
        ).max  # the exact maxval of integer will always be used to represent "nan" or "outofbounds"
        img_quant = np.zeros_like(img, dtype=to_dtype)
        img_quant[oob_mask] = intmax
        img_quant[~oob_mask] = (img_norm[~oob_mask] * (intmax - 1)).astype(to_dtype) + 1

    return img_quant
        logger.debug(
            f"Quantizing {input_img_path=} to {output_img_path=}, {img.min()=:.2f}, {img.max()=:.2f}, {img_quant.min()=:.2f}, {img_quant.max()=:.2f}"
        )

    if (
        output_img_path.suffix == ".png"
        and img_quant.ndim == 3
        and img_quant.shape[-1] == 2
    ):
        # add an empty third channel for compatibility with png formats
        img_quant = np.concatenate(
            [img_quant, np.zeros_like(img_quant[:, :, :1])], axis=2
        )

    return img_quant


def _curlyframe_to_ffmpeg_frametemplate(input_path: Path, as_glob: bool = False):
    newname = re.sub(
        r"\{frame.*:(0\d+)d\}",  # e.g. {frame:06d} or {framenext:06d}
        lambda m: "*" if as_glob else f"%{m.group(1)}d",
        input_path.name,
    )
    return str(input_path.parent / newname)


def unpack_video(
    input_video_path: Path,
    output_frames_path_template: Path,
    ffmpeg: str = "ffmpeg",
    n_cpus: int | None = None,
    loglevel: int = None,
):
    logger.info(
        f"{unpack_video.__name__} {input_video_path=} to {output_frames_path_template=}"
    )
    output_frames_path_template.parent.mkdir(parents=True, exist_ok=True)

    output_path_ffmpeg = _curlyframe_to_ffmpeg_frametemplate(
        output_frames_path_template
    )

    ffmpeg_args = [ffmpeg, "-y", "-hide_banner"]
    
    if loglevel != logging.DEBUG and loglevel != logging.INFO:
        ffmpeg_args.extend(["-loglevel", "error"])
    
    if n_cpus is not None:
        ffmpeg_args.extend(["-threads", str(n_cpus)])
    
    ffmpeg_args.extend(["-i", str(input_video_path), output_path_ffmpeg])
    
    command = " ".join(ffmpeg_args)
    logger.info(
        f"Unpacking {input_video_path=} to {output_frames_path_template=}, {command=}"
    )
    subprocess.check_output(ffmpeg_args)

    if "{framenext" in output_path_ffmpeg:
        for info, path in match_template_paths(output_frames_path_template):
            info["framenext"] = info["frame"] + 1
            next_path = format_template(output_frames_path_template, info)
            shutil.move(path, next_path)


def pack_video(
    input_frames_path: Path,
    output_video_path: Path,
    ffmpeg: str = "ffmpeg",
    n_cpus: int | None = None,
    loglevel: int = None,
):
    logger.info(f"{pack_video.__name__} {input_frames_path=} to {output_video_path=}")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    matched = next(match_template_paths(input_frames_path), None)
    if matched is None:
        raise ValueError(f"No frames found in {input_frames_path=}")
    first = load_any_image(matched[1])
    assert first is not None, f"Failed to load {matched[1]=}"

    dim = first.shape[-1] if first.ndim == 3 else 1
    encoder, pix_fmt = PROPS_TO_ENCODER_PIXFMT[(str(first.dtype), dim)]
    encoder_args = ENCODER_ARGS[encoder]

    input_frames_ffmpeg = _curlyframe_to_ffmpeg_frametemplate(
        input_frames_path, as_glob=True
    )
    
    ffmpeg_args = [ffmpeg, "-y", "-hide_banner"]
    
    if loglevel != logging.DEBUG and loglevel != logging.INFO:
        ffmpeg_args.extend(["-loglevel", "error"])
    
    if n_cpus is not None:
        ffmpeg_args.extend(["-threads", str(n_cpus)])
    
    ffmpeg_args.extend(["-pattern_type", "glob", "-i", input_frames_ffmpeg])
    ffmpeg_args.extend(encoder_args.split())
    ffmpeg_args.extend(["-pix_fmt", pix_fmt, "-an", str(output_video_path)])
    
    command = " ".join(ffmpeg_args)
    logger.info(f"Packing {input_frames_path=} to {output_video_path=}, {command=}")
    subprocess.check_output(ffmpeg_args)


def match_template_paths(
    template: Path,
    match_video_folder: bool = False,
):
    parts = template.parts
    first_curlypart = next((i for i, p in enumerate(parts) if "{" in p), None)
    child_template = "/".join(parts[first_curlypart:])
    search_folder = Path("/".join(parts[:first_curlypart]))

    fmt = Formatter()

    parts = []
    for lit, field, *_ in fmt.parse(child_template):
        parts.append(re.escape(lit))
        if not field:
            continue
        if ":" in field:
            field_name = field.split(":")[0]
            parts.append(rf"(?P<{field_name}>\d+)")
        else:
            parts.append(rf"(?P<{field}>[^/\\]+)")
    regex = "^" + "".join(parts) + "$"
    try:
        regex = re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid regex: {regex=}, {e=}") from e

    def match_to_dict(m: re.Match):
        return {k: int(v) if v.isdigit() else v for k, v in m.groupdict().items()}

    glob_pattern = re.sub(r"\{[^}]*\}", "*", child_template)

    logger.info(f"Searching {search_folder=} for {glob_pattern=}")
    for p in sorted(list(search_folder.rglob(glob_pattern))):
        teststr = str(p.relative_to(search_folder))
        m = regex.match(teststr)
        if m:
            yield match_to_dict(m), p


def pack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
        img_quant = img_orig_to_quant(
):

    logger.debug(
        f"{pack_frameset.__name__} {input_path_template=} to {output_path_template=}"
    )

    output_path_template.parent.mkdir(parents=True, exist_ok=True)

    all_files = list(match_template_paths(input_path_template))
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        img_quant = img_orig_to_quant(
    all_files = list(match_template_paths(input_path_template))
    unpack_channels_last: int | None = None,
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    unpack_channels_last: int | None = None,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    unpack_channels_last: int | None = None,
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        unpack_channels_last=unpack_channels_last,
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    unpack_channels_last: int | None = None,
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)
    img_unquant = img_quant_to_orig(
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

        unpack_channels_last=unpack_channels_last,
    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
    logger.debug(
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        unpack_channels_last=unpack_channels_last,
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
):
    assert input_img_path.suffix == ".png"
    img = load_any_image(input_img_path)

                unpack_channels_last=job["config"].get("unpack_channels_last", None),
    assert np.issubdtype(img.dtype, np.integer), f"{input_img_path=} had {img.dtype=}"

    img_unquant = img_quant_to_orig(
        img,
        min_orig_val,
        max_orig_val,
        to_dtype=to_dtype,
        quantize_method=quantize_method,
        unpack_channels_last=unpack_channels_last,
    )

    if quantize_method != QuantizeMethod.CHECKBOUNDS:
                unpack_channels_last=job["config"].get("unpack_channels_last", None),
        isnan = img == np.iinfo(img.dtype).max
        img_unquant[isnan] = np.nan

    logger.debug(
        f"Unquantizing {input_img_path=} to {output_img_path=}, {img_unquant.min()=:.2f}, {img_unquant.max()=:.2f}"
    )

    save_any_image(img_unquant, output_img_path)


def unpack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        unpack_channels_last=unpack_channels_last,
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    unpack_channels_last: int | None = None,
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)
    img_unquant = img_quant_to_orig(
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

        unpack_channels_last=unpack_channels_last,
    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
    logger.debug(
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        unpack_channels_last=unpack_channels_last,
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
):
    assert input_img_path.suffix == ".png"
    img = load_any_image(input_img_path)

    assert np.issubdtype(img.dtype, np.integer), f"{input_img_path=} had {img.dtype=}"

    img_unquant = img_quant_to_orig(
        img,
        min_orig_val,
        max_orig_val,
        to_dtype=to_dtype,
        quantize_method=quantize_method,
        unpack_channels_last=unpack_channels_last,
    )

    if quantize_method != QuantizeMethod.CHECKBOUNDS:
        isnan = img == np.iinfo(img.dtype).max
        img_unquant[isnan] = np.nan

    logger.debug(
        f"Unquantizing {input_img_path=} to {output_img_path=}, {img_unquant.min()=:.2f}, {img_unquant.max()=:.2f}"
    )

    save_any_image(img_unquant, output_img_path)


def unpack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
            unpack_channels_last=unpack_channels_last,
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        unpack_channels_last=unpack_channels_last,
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    unpack_channels_last: int | None = None,
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)
    img_unquant = img_quant_to_orig(
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

        unpack_channels_last=unpack_channels_last,
    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
    logger.debug(
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    unpack_channels_last: int | None = None,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
    img_unquant = img_quant_to_orig(
        save_any_image(img_quant, output_path)
    if len(all_files) == 0:
        raise ValueError(f"No frames found in {input_path_template=}")

    for frame_info, frame_input_path in all_files:
        unpack_channels_last=unpack_channels_last,
        output_path = format_template(output_path_template, frame_info)

        quantize_frame(
            frame_input_path,
            output_path,
            to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            out_of_bounds_method=out_of_bounds_method,
        )
        save_any_image(img_quant, output_path)

def unquantize_frame(
    input_img_path: Path,
    output_img_path: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
):
    assert input_img_path.suffix == ".png"
    img = load_any_image(input_img_path)

    assert np.issubdtype(img.dtype, np.integer), f"{input_img_path=} had {img.dtype=}"

    img_unquant = img_quant_to_orig(
        img,
        min_orig_val,
        max_orig_val,
        to_dtype=to_dtype,
        quantize_method=quantize_method,
        unpack_channels_last=unpack_channels_last,
    )

    if quantize_method != QuantizeMethod.CHECKBOUNDS:
        isnan = img == np.iinfo(img.dtype).max
        img_unquant[isnan] = np.nan

    logger.debug(
        f"Unquantizing {input_img_path=} to {output_img_path=}, {img_unquant.min()=:.2f}, {img_unquant.max()=:.2f}"
    )

    save_any_image(img_unquant, output_img_path)


def unpack_frameset(
    input_path_template: Path,
    output_path_template: Path,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    min_orig_val: float,
    max_orig_val: float,
):
    assert "{" in input_path_template.name, (
        f"Input path must contain a template: {input_path_template=}"
    )
    assert "{" in output_path_template.name, (
        f"Output path must contain a template: {output_path_template=}"
    )
    output_path_template.parent.mkdir(parents=True, exist_ok=True)
    all_files = match_template_paths(input_path_template)

    for frame_info, frame_input_path in all_files:
        unquantize_frame(
            frame_input_path,
            output_img_path=format_template(output_path_template, frame_info),
            to_dtype=to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
            unpack_channels_last=unpack_channels_last,
        )


def format_template(template: Path, vals: dict, allow_missing: list[str] | None = None):
    def replace_func(match):
        full_spec = match.group(1)
        key = full_spec.split(":")[0]
        if key in vals:
            try:
                return ("{" + full_spec + "}").format(**{key: vals[key]})
            except ValueError as e:
                raise ValueError(f"Invalid {full_spec=} for {key=} {vals[key]=} in {template=}, {e=}") from e
        elif allow_missing and key not in allow_missing:
            raise ValueError(f"Missing {key=} in {vals=} for {template=}, {allow_missing=}")
        
        return match.group(0)

    res = re.sub(r"\{([^}]+)\}", replace_func, str(template))

    if isinstance(template, Path):
        return Path(res)
    return res


def find_jobs(
    input_template: Path,
    output_template: Path,
    gt_type: str,
    subset: dict,
    extra_job_args: dict,
    match_video_folder: bool = False,
    lazy: bool = False,
):
    input_template = format_template(input_template, {"gt_type": gt_type})

    if match_video_folder and "{frame" in input_template.parts[-1]:
        search_template = input_template.parent
        input_template_extra = input_template.parts[-1]
    else:
        search_template = input_template
        input_template_extra = None
        
    paths = match_template_paths(search_template)

    skipped_for_lazy = 0
    jobs = []
    for vid_info, vid_input_path in paths:

        vid_info["gt_type"] = gt_type

        if subset and not all(
            k not in vid_info or vid_info[k] == v for k, v in subset.items()
        ):
            continue

        output_path = format_template(output_template, vid_info, allow_missing=[])
        if lazy and output_path.exists():
            skipped_for_lazy += 1
            continue

        if input_template_extra:
            extra = format_template(input_template_extra, vid_info)
            vid_input_path = vid_input_path / extra

        jobs.append(
            {
                "input_path": vid_input_path,
                "output_path": output_path,
                **extra_job_args,
            }
        )

    if len(jobs) == 0 and skipped_for_lazy == 0:
        raise ValueError(f"No jobs found for {input_template}")
    msg = f"Found {len(jobs)} jobs for {input_template} -> {output_template}"
    if skipped_for_lazy > 0:
        msg += f", skipped {skipped_for_lazy} due to --lazy flag"
    logger.info(msg)

    return jobs


def _parse_k_equals_v_strs(k_equals_v_strs: list[str] | str | None):
    if k_equals_v_strs is None:
        return {}
    elif isinstance(k_equals_v_strs, str):
        k_equals_v_strs = k_equals_v_strs.split(" ")

    args = {}
    for arg in k_equals_v_strs:
        parts = arg.split("=")
        if len(parts) != 2:
            raise ValueError(f"Invalid {arg=}, had {len(parts)=}")
        k, v = parts
        args[k] = v
    return args


def _calculate_config_precision(method, low, high, dtype):
    pass


def process_video_job(job: dict):

    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])

    tmp_folder = job["tmp_folder"]
    if tmp_folder is not None:
        out_str = str(output_path)
        out_str = (
            out_str.replace("{", "")
            .replace("}", "")
            .replace(":", "")
            .replace("_", "-")
            .replace("/", "_")
        )
        tmp_path = tmp_folder / out_str
        tmp_path.mkdir(parents=True, exist_ok=False)
    else:
        tmp_path = None

    logger.info(f"Processing {input_path} -> {tmp_path}{output_path}")

    match input_path.suffix, output_path.suffix:
        case ".png", ".mkv":
            pack_video(input_path, output_path, n_cpus=job.get("cpus_per_worker"), loglevel=job.get("loglevel"))
        case _, ".mkv":
            tmp_template = tmp_path / "{frame:06d}.png"
            pack_frameset(
                input_path,
                tmp_template,
                to_dtype=DTYPE_MAP[job["config"]["pack_dtype"]],
                quantize_method=QuantizeMethod.from_str(
                    job["config"]["quantize_method"]
                ),
                min_orig_val=float(job["config"]["min_orig_val"]),
                max_orig_val=float(job["config"]["max_orig_val"]),
                out_of_bounds_method=job["config"]["out_of_bounds_method"],
            )
            pack_video(tmp_template, output_path, n_cpus=job.get("cpus_per_worker"), loglevel=job.get("loglevel"))
        case ".mkv", ".png":
            unpack_video(input_path, output_path, n_cpus=job.get("cpus_per_worker"), loglevel=job.get("loglevel"))
        case '.png' | '.jpg' | '.jpeg' | '.npy', ".png":
            pack_frameset(
                input_path,
                output_path,
                to_dtype=DTYPE_MAP[job["config"]["pack_dtype"]],
                quantize_method=QuantizeMethod.from_str(
                    job["config"]["quantize_method"]
                ),
                min_orig_val=float(job["config"]["min_orig_val"]),
                max_orig_val=float(job["config"]["max_orig_val"]),
                out_of_bounds_method=job["config"]["out_of_bounds_method"],
            )
        case ".png", _:
            unpack_frameset(
                input_path,
                output_path,
                to_dtype=DTYPE_MAP[job["config"]["unpack_dtype"]],
                quantize_method=QuantizeMethod.from_str(
                    job["config"]["quantize_method"]
                ),
                min_orig_val=float(job["config"]["min_orig_val"]),
                max_orig_val=float(job["config"]["max_orig_val"]),
                unpack_channels_last=job["config"].get("unpack_channels_last", None),
            )
        case ".mkv", _:
            tmp_frames = tmp_path / "{frame:06d}.png"
            unpack_video(input_path, tmp_frames, n_cpus=job.get("cpus_per_worker"), loglevel=job.get("loglevel"))
            unpack_frameset(
                tmp_frames,
                output_path,
                to_dtype=DTYPE_MAP[job["config"]["unpack_dtype"]],
                quantize_method=QuantizeMethod.from_str(
                    job["config"]["quantize_method"]
                ),
                min_orig_val=float(job["config"]["min_orig_val"]),
                max_orig_val=float(job["config"]["max_orig_val"]),
                unpack_channels_last=job["config"].get("unpack_channels_last", None),
            )
        case ".txt", ".npy":
            data = np.loadtxt(input_path)
            assert "{" not in str(output_path), output_path
            np.save(output_path, data)
            assert output_path.exists(), f"Failed to save {output_path=}"
        case ".npy", ".txt":
            data = np.load(input_path)
            assert "{" not in str(output_path), output_path
            np.savetxt(output_path, data)
            assert output_path.exists(), f"Failed to save {output_path=}"
        case x, y if x == y:
            shutil.copy(input_path, output_path)
        case _:
            raise ValueError(f"Invalid {input_path.suffix=} {output_path.suffix=}")

    if tmp_path is not None:
        shutil.rmtree(tmp_path)


def wait_jobs(launched_jobs, pbar):
    """Wait for a list of submitted jobs to complete, checking periodically."""
    finished_jobs = set()
    crashed_jobs = []
    while len(finished_jobs) < len(launched_jobs):
        for j in launched_jobs:
            if j.job_id in finished_jobs:
                continue
            if j.state in ["PENDING", "RUNNING"]:
                continue

            try:
                result = j.result()
                pbar.update(1)
                pbar.set_description(f"Job {j.job_id} completed successfully")
            except Exception as e:
                msg = f"Job {j.job_id} failed with error: {e}"
                pbar.update(1)
                pbar.set_description(msg)
                logger.error(msg)
                crashed_jobs.append(j)
            finished_jobs.add(j.job_id)

        time.sleep(1)

    return crashed_jobs


def execute_jobs(
    log_folder: Path,
    func: Callable,
    jobs: list[dict],
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    n_workers: int | None,
    slurm_args: list[str],
    cpus_per_worker: int | None = None,
):
    logger.info(f"Executing {len(jobs)} jobs with {paralell_mode=} {n_workers=}")

    if n_workers is None:
        for job in jobs:
            func(job)
        return

    match paralell_mode:
        case "multiprocess":
            with multiprocessing.Pool(n_workers) as pool:
                pool.map(func, jobs)
        case "slurm":
            if submitit is None:
                raise ValueError(
                    "submitit is not installed. Please install, or install cvdpack[slurm] optional extras"
                )
            log_folder.mkdir(parents=True, exist_ok=True)
            executor = submitit.AutoExecutor(
                folder=log_folder,
            )
            executor.update_parameters(
                slurm_mem_gb=4,
                slurm_cpus_per_task=cpus_per_worker or 4,
                slurm_time=60,
                slurm_array_parallelism=n_workers,
            )
            if slurm_args:
                slurm_args = _parse_k_equals_v_strs(slurm_args)
                executor.update_parameters(**slurm_args)

            pbar = tqdm(total=len(jobs), desc="Running jobs")
            crashed = []
            for i in range(0, len(jobs), SLURM_ARRAY_MAX):
                launched = executor.map_array(
                    func, jobs[i : i + SLURM_ARRAY_MAX]
                )
                crashed += wait_jobs(launched, pbar)
            if len(crashed) > 0:
                raise ValueError(
                    f"{len(crashed)} jobs crashed, dataset is likely not safe to use. "
                    f"Please check {log_folder} for ID_log.err and ID_log.out for each ID in {crashed}"
                )
        case _:
            raise ValueError(f"Invalid {paralell_mode=}")

def decide_dataset_job_templates(
    input_folder: Path,
    output_folder: Path,
    datatype_conf: dict,
    steps: list[str] | None,
    mode: Literal["pack", "unpack"],
):
    """
    By default, we always go to/from templates in the config

    If the user restricts `steps`, we may then need to go to/from intermediate vals like quantized pngs
    """

    if mode == "pack":
        default_src = Path(datatype_conf["original_path_template"])
        default_dest = Path(datatype_conf["packed_path_template"])
    elif mode == "unpack":
        default_src = Path(datatype_conf["packed_path_template"])
        default_dest = Path(datatype_conf["original_path_template"])
    else:
        raise ValueError(f"Invalid {mode=}")

    inp = default_src
    out = default_dest

    logger.debug(f"{decide_dataset_job_templates.__name__} {default_src.suffix} -> {default_dest.suffix} {steps=} {mode=}")

    if steps is None:
        logger.debug(f"No steps specified, using default {inp=} {out=}")
        return input_folder / inp, output_folder / out
    if len(steps) == 0:
        raise ValueError("User specified empty --steps, there is no work to be done?")

    if mode == "pack" and default_dest.suffix == ".mkv":
        if steps == ["quantize"]:
            # we are not actually going all the way to video, we are stopping at pngs
            out = default_src.with_suffix(".png")
            logger.debug(f"Changed from {default_dest=} to {out=} due to {mode=} {steps=}")
        elif steps == ["pack_video"]:
            # we are starting from already quantized pngs
            inp = default_src.with_suffix(".png")
            logger.debug(f"Changed from {default_src=} to {inp=} due to {mode=} {steps=}")
        else:
            raise ValueError(f"Unhandled {steps=} for {mode=} {default_dest=}")
    elif mode == "unpack" and default_src.suffix == ".mkv":
        if steps == ["unquantize"]: 
            # we are not actually going all the way to video, we are stopping at pngs
            inp = default_dest.with_suffix(".png")
            logger.debug(f"Changed from {default_src=} to {inp=} due to {mode=} {steps=}")
        elif steps == ["unpack_video"]: 
            # unpack the video, but stop before .npy, leave pngs instead
            if default_dest.suffix == ".npy":
                out = default_dest.with_suffix(".png")
                logger.debug(f"Changed from {default_dest=} to {out=} due to {mode=} {steps=}")
        else:
            raise ValueError(f"Unhandled {steps=} for {mode=} {default_src=}")
    elif (
        (mode == "pack" and default_src.suffix == ".txt" and steps == ["pack_video"])
        or (mode == "unpack" and default_src.suffix == ".npy" and steps == ["unpack_video"])
    ):
        inp = inp.with_suffix(default_dest.suffix)
        logger.debug(
            f"Changed from {default_src=} to {inp=} due to {mode=} {steps=}, "
            "packing/unpacking from txt happens in quantize/unquantize not video pack"
        )
    else:
        logger.debug(f"{decide_dataset_job_templates=} didnt match any cases, using {inp=} {out=}")

    return input_folder / inp, output_folder / out


def pack_dataset(
    input_folder: Path,
    output_folder: Path,
    steps: list[str] | None,
    config_path: Path | None,
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: list[str],
    n_workers: int,
    tmp_folder: Path,
    subset: list[str],
    lazy: bool,
    cpus_per_worker: int | None = None,
    loglevel: int = None,
):
    if config_path is None:
        config_path = input_folder / "cvdpack.json"
    if not config_path.exists():
        raise ValueError(f"Could not find {config_path=}")
    with config_path.open("r") as f:
        config = json.load(f)

    subset = _parse_k_equals_v_strs(subset)

    jobs = []
    for gt_type, datatype_conf in config["data_types"].items():

        input_template, output_template = decide_dataset_job_templates(
            input_folder, output_folder, datatype_conf, steps, "pack",
        )

        jobs.extend(
            find_jobs(
                input_template=input_template,
                output_template=output_template,
                gt_type=gt_type,
                subset=subset,
                extra_job_args={"tmp_folder": tmp_folder, "config": datatype_conf, "cpus_per_worker": cpus_per_worker, "loglevel": loglevel},
                lazy=lazy,
                match_video_folder=True,
            )
        )

    for gt_type, gt_conf in config["data_types"].items():
        quantize_method = gt_conf.get("quantize_method", "NONE")
        if quantize_method in ["NONE", "CHECKBOUNDS"]:
            continue
        gt_conf["quantize_precision"] = _calculate_config_precision(
            method=QuantizeMethod.from_str(quantize_method),
            low=float(gt_conf["min_orig_val"]),
            high=float(gt_conf["max_orig_val"]),
            dtype=gt_conf["pack_dtype"],
        )

    execute_jobs(
        log_folder=output_folder / "logs",
        func=process_video_job,
        jobs=jobs,
        paralell_mode=paralell_mode,
        n_workers=n_workers,
        slurm_args=slurm_args,
        cpus_per_worker=cpus_per_worker,
    )


def unpack_dataset(
    input_folder: Path,
    output_folder: Path,
    steps: list[str] | None,
    config_path: Path | None,
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: list[str],
    n_workers: int,
    subset: list[str],
    tmp_folder: Path,
    lazy: bool,
    cpus_per_worker: int | None = None,
    loglevel: int = None,
):
    if config_path is None:
        config_path = input_folder / "cvdpack.json"
    if not config_path.exists():
        raise ValueError(f"Could not find {config_path=}")
    with config_path.open("r") as f:
        config = json.load(f)

    subset = _parse_k_equals_v_strs(subset)

    jobs = []
    for datatype_conf in config["data_types"]:

        search_input_template, search_output_template = decide_dataset_job_templates(
            input_folder, output_folder, datatype_conf, steps, mode="unpack",
        )

        jobs.extend(
            find_jobs(
                input_template=search_input_template,
                output_template=search_output_template,
                subset=subset,
                extra_job_args={"tmp_folder": tmp_folder, "config": datatype_conf, "cpus_per_worker": cpus_per_worker, "loglevel": loglevel},
                lazy=lazy,
                match_video_folder=True,
            )
        )

    execute_jobs(
        log_folder=output_folder / "logs",
        func=process_video_job,
        jobs=jobs,
        paralell_mode=paralell_mode,
        n_workers=n_workers,
        slurm_args=slurm_args,
        cpus_per_worker=cpus_per_worker,
    )


def validate_args(args: argparse.Namespace):
    if args.config is not None and args.config.parts[0] == "presets":
        args.config = Path(__file__).parent / args.config

    if args.parallel_mode != "none" and not args.action.endswith("_dataset"):
        raise ValueError(
            f"--parallel_mode {args.parallel_mode=} only applies to paralellism over videos, not {args.action=}."
            " per-frame paralellism is not currently supported"
        )

    avoids_ffmpeg = (
        args.steps is not None
        and len(set(args.steps).intersection({"pack_video", "unpack_video"})) == 0
    )

    if not avoids_ffmpeg:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError(
                "ffmpeg is required for video operations but was not found. "
                "Please install ffmpeg and ensure it's available in your PATH."
            )

    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        type=str,
        choices=[
            "pack_frames",
            "unpack_frames",
            "pack_dataset",
            "unpack_dataset",
        ],
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--steps", 
        type=str, 
        default=None, 
        nargs="*", 
        choices=["quantize", "pack_video", "unpack_video", "unquantize"]
    )

    # frames level configs - valid only for level=frameset
    parser.add_argument("--to_dtype", type=str, choices=DTYPE_MAP.keys(), default=None)
    parser.add_argument(
        "--quantize_method",
        type=QuantizeMethod.from_str,
        default=None,
        choices=list(QuantizeMethod),
    )
    parser.add_argument("--min_orig_val", type=float, default=None)
    parser.add_argument("--max_orig_val", type=float, default=None)
    parser.add_argument(
        "--out_of_bounds_method",
        type=str,
        choices=["nan", "nan_warn", "error"],
        default="nan_warn",
    )

    # scene level configs - valid only for level="scene"
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="multiprocess",
        choices=["multiprocess", "slurm", "none"],
        help="What parallelization method to use when n_workers is specified. `slurm` requires cvdpack[slurm] optional dependencies.",
    )
    parser.add_argument(
        "--slurm_args",
        type=str,
        default=None,
        nargs="*",
        help="Must be space separated and use key=value format. Passed to submitit executor.update_parameters",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None, help="Number of jobs to run in parallel."
    )
    parser.add_argument(
        "--cpus_per_worker", type=int, default=None, help="Number of CPUs per worker for slurm and ffmpeg threads."
    )
    parser.add_argument(
        "--subset",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Restricts the pack/unpack to only operate on some scenes/gt/cameras. "
            "Must be list of key=value pairs, where keys match the template placeholders. "
            "e.g. scene=xyz, cam=left, etc."
        ),
    )
    parser.add_argument("--tmp_folder", type=Path, default=None)
    parser.add_argument("--lazy", action="store_true", default=False)

    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )

    return validate_args(parser.parse_args())


def format_for_json(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main():

    start_time = time.time()

    args = parse_args()
    
    # Configure logging with a console handler
    logging.basicConfig(
        level=args.loglevel,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()]
    )
    logger.setLevel(args.loglevel)

    out_suffix = (
        args.output.suffix if not args.output.is_dir() else None
    )
    match args.action, out_suffix:
        case "pack_frames", ".png":
            pack_frameset(
                args.input,
                args.output,
                args.to_dtype,
                args.quantize_method,
                args.min_orig_val,
                args.max_orig_val,
                args.out_of_bounds_method,
            )
        case "pack_frames", ".mkv":
            pack_video(
                args.input,
                args.output,
                n_cpus=args.cpus_per_worker,
                loglevel=args.loglevel,
            )
        case "unpack_frames", ".png":
            unpack_frameset(
                args.input,
                args.output,
                args.to_dtype,
                args.quantize_method,
                args.min_orig_val,
                args.max_orig_val,
            )
        case "unpack_frames", ".mkv":
            unpack_video(
                args.input,
                args.output,
                n_cpus=args.cpus_per_worker,
                loglevel=args.loglevel,
            )
        case "pack_dataset", _:
            if not args.input.is_dir():
                raise ValueError(f"pack_dataset requires input to be a directory: {args.input=}")
            pack_dataset(
                args.input,
                args.output,
                args.steps,
                args.config,
                args.parallel_mode,
                args.slurm_args,
                args.n_workers,
                args.tmp_folder,
                args.subset,
                args.lazy,
                args.cpus_per_worker,
                args.loglevel,
            )
        case "unpack_dataset", _:
            if not args.input.is_dir():
                raise ValueError(f"unpack_dataset requires input to be a directory: {args.input=}")
            unpack_dataset(
                args.input,
                args.output,
                args.steps,
                args.config,
                args.parallel_mode,
                args.slurm_args,
                args.n_workers,
                args.subset,
                args.tmp_folder,
                args.lazy,
                args.cpus_per_worker,
                args.loglevel,
            )
        case _:
            raise ValueError(f"Invalid {args.action=}")

    if args.config is None:
        return

    with args.config.open("r") as f:
        config = json.load(f)

    if args.action == "pack_frames" and args.output.suffix == ".png":
        config["data_types"].append(
            {
                "original": args.input,
                "packed": args.output,
                "min_orig_val": args.min_orig_val,
                "max_orig_val": args.max_orig_val,
                "quantize_method": args.quantize_method,
                "out_of_bounds_method": args.out_of_bounds_method,
                "pack_dtype": args.to_dtype,
            }
        )
    elif args.action.endswith("_dataset"):
        config["metadata"]["original_folder"] = str(args.input)
        config["metadata"]["packed_folder"] = str(args.output)

    config["metadata"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["metadata"]["cvdpack_version"] = __version__
    config["metadata"]["args"] = vars(args)
    config["metadata"]["pack_runtime"] = time.time() - start_time

    if args.action.endswith("_dataset"):
        with (args.output / "cvdpack.json").open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)
    elif args.config is not None and not str(args.config).startswith("presets/"):
        with args.config.open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)

    print(f"Completed {args.action} for result {args.output} in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
