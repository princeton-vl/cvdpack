import argparse
import copy
import json
import logging
import multiprocessing
import os
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


ENCODER_ARGS = {
    "ffv1": os.environ.get(
        "CVDPACK_FFV1_ARGS", "-c:v ffv1 -level 3 -g 1 -slices 4 -threads 4 -slicecrc 1"
    ),
    "libx265": os.environ.get(
        "CVDPACK_LIBX265_ARGS", "-c:v libx265 -x265-params lossless=1 -preset slow"
    ),
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
        case ".png" | ".jpg" | ".jpeg":
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
        case ((".png" | ".jpg" | ".jpeg"), np.uint8 | np.uint16):
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
        # case QuantizeMethod.SYM_SQRT:
        #    img_norm = sym_sqrt(img)
        #    sqrt_min = sym_sqrt(min_orig_val)
        #    sqrt_max = sym_sqrt(max_orig_val)
        #    img_norm = (img_norm - sqrt_min) / (sqrt_max - sqrt_min)
        #    return img_norm
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
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
    unpack_channels_last: int | None = None,
) -> np.ndarray:
    assert np.issubdtype(img.dtype, np.unsignedinteger), f"{img.dtype=}"
    from_max = np.iinfo(img.dtype).max - 1  # exact maxint val is used for nan

    match quantize_method:
        case QuantizeMethod.CHECKBOUNDS:
            img = img.astype(to_dtype)
        case QuantizeMethod.LINEAR:
            img_norm = img.astype(np.float64) / from_max
            img_orig = img_norm * (max_orig_val - min_orig_val) + min_orig_val
            img = img_orig.astype(to_dtype)
        case QuantizeMethod.INV:
            img_norm = img.astype(np.float64) / from_max
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_unmap = img_norm * (max_norm - min_norm) + min_norm
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

    if logger.isEnabledFor(logging.DEBUG):
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
    if "{frame}" in str(input_path):
        # note RE this warning - we could potentially map {frame} to %04d for ffmpeg, but:
        # (1) it is hard to guess the num digits and
        # (2) this is difficult for cases like TartanAir flow where paths contain both {frame:06d} and {framenext:06d}
        raise ValueError(
            "Direct use of {{frame}} as a template is banned - you must use {{frame:04d}} for some integer. "
            "this is because frame without 0 padding will not be sorted properly when globbed to form a video"
        )

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

    if loglevel != logging.DEBUG:
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

    return command


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

    if loglevel != logging.DEBUG:
        ffmpeg_args.extend(["-loglevel", "error"])
        if encoder == "libx265":
            encoder_args += " -x265-params log-level=quiet"

    if n_cpus is not None:
        ffmpeg_args.extend(["-threads", str(n_cpus)])

    ffmpeg_args.extend(["-pattern_type", "glob", "-i", input_frames_ffmpeg])
    ffmpeg_args.extend(encoder_args.split())
    ffmpeg_args.extend(["-pix_fmt", pix_fmt, "-an", str(output_video_path)])

    command = " ".join(ffmpeg_args)
    logger.info(f"Packing {input_frames_path=} to {output_video_path=}, {command=}")
    subprocess.check_output(ffmpeg_args)

    return command


def match_template_paths(
    template: Path,
    match_video_folder: bool = False,
) -> list[tuple[dict, Path]]:
    parts = template.parts
    first_curlypart = next((i for i, p in enumerate(parts) if "{" in p), None)
    child_template = "/".join(parts[first_curlypart:])
    search_folder = Path("/".join(parts[:first_curlypart]))

    fmt = Formatter()

    parts = []
    for lit, field, conv, _ in fmt.parse(child_template):
        if "*" in lit:
            lit_parts = lit.split("*")
            for i, part in enumerate(lit_parts):
                if i > 0:
                    parts.append(r"[^/\\]*")
                parts.append(re.escape(part))
        else:
            parts.append(re.escape(lit))

        if not field:
            continue

        if conv.endswith("d"):
            parts.append(rf"(?P<{field}>\d+)")
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

    files = sorted(list(search_folder.rglob(glob_pattern)))
    logger.debug(
        f"{search_folder=} had {len(files)} files matching {glob_pattern=}, testing against {regex=}"
    )
    for p in files:
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
    out_of_bounds_method: Literal["nan", "nan_warn", "error"] = "nan_warn",
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
    unpack_channels_last: int | None = None,
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
                raise ValueError(
                    f"Invalid {full_spec=} for {key=} {vals[key]=} in {template=}, {e=}"
                ) from e
        elif allow_missing and key not in allow_missing:
            raise ValueError(
                f"Missing {key=} in {vals=} for {template=}, {allow_missing=}"
            )

        return match.group(0)

    res = re.sub(r"\{([^}]+)\}", replace_func, str(template))

    if isinstance(template, Path):
        return Path(res)
    return res


def find_jobs(
    input_template: Path,
    output_template: Path,
    gt_type: str,
    subset: dict | None,
    extra_job_args: dict,
    match_video_folder: bool = False,
    lazy: bool = False,
) -> list[dict]:
    input_template = format_template(input_template, {"gt_type": gt_type})

    if match_video_folder and "{frame" in input_template.parts[-1]:
        search_template = input_template.parent
        input_template_extra = input_template.parts[-1]
    else:
        search_template = input_template
        input_template_extra = None

    paths = list(match_template_paths(search_template))
    paths, skipped_for_subset = filter_files_by_subset_dict(paths, subset)

    skipped_for_lazy = 0
    jobs = []
    for vid_info, vid_input_path in paths:
        vid_info["gt_type"] = gt_type

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

    if len(jobs) == 0 and skipped_for_lazy == 0 and skipped_for_subset == 0:
        raise ValueError(f"No jobs found for {input_template}")
    msg = f"Found {len(jobs)} jobs for {input_template} -> {output_template}"
    if skipped_for_lazy > 0:
        msg += f", skipped {skipped_for_lazy} due to --lazy flag"
    if skipped_for_subset > 0:
        msg += f", skipped {skipped_for_subset} due to --subset flag"
    logger.info(msg)

    return jobs


def _parse_k_equals_v_strs(k_equals_v_strs: list[str] | None):
    if k_equals_v_strs is None:
        return None

    logger.debug(f"_parse_k_equals_v_strs received: {k_equals_v_strs}")
    args = {}
    for arg in k_equals_v_strs:
        parts = arg.split("=")
        if len(parts) != 2:
            raise ValueError(f"Invalid {arg=}, had {len(parts)=}")
        k, v = parts
        if "," in v:
            v = list(v.split(","))
        args[k] = v
    return args


def _calculate_config_precision(method, low, high, dtype):
    pass


def process_video_job(job: dict):
    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])

    tmp_path = None

    def make_tmp_folder(exists_ok: bool = False):
        tmp_folder = job["tmp_folder"]
        out_str = str(output_path)
        out_str = (
            out_str.replace("{", "")
            .replace("}", "")
            .replace(":", "")
            .replace("_", "-")
            .replace("/", "_")
        )

        nonlocal tmp_path
        tmp_path = tmp_folder / out_str

        logger.debug(
            f"Making {tmp_path=} for {input_path=} -> {output_path=}, {tmp_path.exists()=}"
        )

        tmp_path.mkdir(parents=True, exist_ok=exists_ok)
        return tmp_path

    logger.info(f"Processing {input_path} -> {output_path}")

    metadata_commands = []

    match input_path.suffix, output_path.suffix:
        case ".png", ".mkv":
            command = pack_video(
                input_path,
                output_path,
                n_cpus=job.get("cpus_per_worker"),
                loglevel=job.get("loglevel"),
            )
            metadata_commands.append(command)
        case _, ".mkv":
            tmp_template = make_tmp_folder() / "{frame:06d}.png"
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
            command = pack_video(
                tmp_template,
                output_path,
                n_cpus=job.get("cpus_per_worker"),
                loglevel=job.get("loglevel"),
            )
            metadata_commands.append(command)
        case ".mkv", ".png":
            command = unpack_video(
                input_path,
                output_path,
                n_cpus=job.get("cpus_per_worker"),
                loglevel=job.get("loglevel"),
            )
            metadata_commands.append(command)
        case ".png" | ".jpg" | ".jpeg" | ".npy", ".png":
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
            tmp_frames = make_tmp_folder() / "{frame:06d}.png"
            command = unpack_video(
                input_path,
                tmp_frames,
                n_cpus=job.get("cpus_per_worker"),
                loglevel=job.get("loglevel"),
            )
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
            metadata_commands.append(command)
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

    return {
        "commands": metadata_commands,
    }


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
                msg = f"Job {j.job_id} completed successfully with {result=}"
            except Exception as e:
                msg = f"Job {j.job_id} failed with error: {e}"
                logger.error(msg)
                crashed_jobs.append(j)

            pbar.update(1)
            pbar.set_description(msg)
            finished_jobs.add(j.job_id)

        time.sleep(1)

    return crashed_jobs


def execute_jobs(
    log_folder: Path,
    func: Callable,
    jobs: list[dict],
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    n_workers: int | None,
    slurm_args: dict | None,
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
            if slurm_args is not None:
                executor.update_parameters(**slurm_args)

            pbar = tqdm(total=len(jobs), desc="Running jobs")
            crashed = []
            for i in range(0, len(jobs), SLURM_ARRAY_MAX):
                launched = executor.map_array(func, jobs[i : i + SLURM_ARRAY_MAX])
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

    assert isinstance(datatype_conf, dict), f"Invalid {datatype_conf=}"

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

    logger.debug(
        f"{decide_dataset_job_templates.__name__} {default_src.suffix} -> {default_dest.suffix} {steps=} {mode=}"
    )

    if steps is None:
        logger.debug(f"No steps specified, using default {inp=} {out=}")
        return input_folder / inp, output_folder / out
    if len(steps) == 0:
        raise ValueError("User specified empty --steps, there is no work to be done?")

    if mode == "pack" and default_dest.suffix == ".mkv":
        if steps == ["quantize"]:
            # we are not actually going all the way to video, we are stopping at pngs
            out = default_src.with_suffix(".png")
            logger.debug(
                f"Changed from {default_dest=} to {out=} due to {mode=} {steps=}"
            )
        elif steps == ["pack_video"]:
            # we are starting from already quantized pngs
            inp = default_src.with_suffix(".png")
            logger.debug(
                f"Changed from {default_src=} to {inp=} due to {mode=} {steps=}"
            )
        else:
            raise ValueError(f"Unhandled {steps=} for {mode=} {default_dest=}")
    elif mode == "unpack" and default_src.suffix == ".mkv":
        if steps == ["unquantize"]:
            # we are not actually going all the way to video, we are stopping at pngs
            inp = default_dest.with_suffix(".png")
            logger.debug(
                f"Changed from {default_src=} to {inp=} due to {mode=} {steps=}"
            )
        elif steps == ["unpack_video"]:
            # unpack the video, but stop before .npy, leave pngs instead
            if default_dest.suffix == ".npy":
                out = default_dest.with_suffix(".png")
                logger.debug(
                    f"Changed from {default_dest=} to {out=} due to {mode=} {steps=}"
                )
        else:
            raise ValueError(f"Unhandled {steps=} for {mode=} {default_src=}")
    elif mode == "pack" and default_src.suffix == ".txt" and steps == ["pack_video"]:
        inp = inp.with_suffix(default_dest.suffix)
        logger.debug(
            f"Changed from {default_src=} to {inp=} due to {mode=} {steps=}, "
            "packing from txt->npy happens in quantize/unquantize not video pack"
        )
    elif (
        mode == "unpack" and default_src.suffix == ".npy" and steps == ["unpack_video"]
    ):
        out = out.with_suffix(default_src.suffix)
        logger.debug(
            f"Changed from {default_dest=} to {out=} due to {mode=} {steps=}, "
            "unpacking from npy->txt happens in quantize/unquantize not video unpack"
        )
    else:
        logger.debug(
            f"{decide_dataset_job_templates=} didnt match any cases, using {inp=} {out=}"
        )

    return input_folder / inp, output_folder / out


def pack_dataset(
    input_folder: Path,
    output_folder: Path,
    steps: list[str] | None,
    config: dict | None,
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: dict | None,
    n_workers: int,
    tmp_folder: Path,
    subset: dict | None,
    lazy: bool,
    cpus_per_worker: int | None = None,
    loglevel: int = None,
):
    if tmp_folder is not None and tmp_folder.exists():
        raise ValueError(
            f"--tmp_folder {tmp_folder} already exists, please use a different one or consider deleting it"
        )

    if config is None:
        raise ValueError(
            "pack_dataset requires a config, must use --config "
            "or use an --input containing a cvdpack.json"
        )

    jobs = []
    for gt_type, datatype_conf in config["data_types"].items():
        input_template, output_template = decide_dataset_job_templates(
            input_folder,
            output_folder,
            datatype_conf,
            steps,
            "pack",
        )

        jobs.extend(
            find_jobs(
                input_template=input_template,
                output_template=output_template,
                gt_type=gt_type,
                subset=subset,
                extra_job_args={
                    "tmp_folder": tmp_folder,
                    "config": datatype_conf,
                    "cpus_per_worker": cpus_per_worker,
                    "loglevel": loglevel,
                },
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
    config: dict | None,
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: dict | None,
    n_workers: int,
    subset: dict | None,
    tmp_folder: Path,
    lazy: bool,
    cpus_per_worker: int | None = None,
    loglevel: int = None,
):
    if config is None:
        raise ValueError(
            "unpack_dataset requires a config, must use --config "
            "or use an --input containing a cvdpack.json"
        )

    if tmp_folder is not None and tmp_folder.exists():
        raise ValueError(
            f"--tmp_folder {tmp_folder} already exists, please use a different one or consider deleting it"
        )

    jobs = []
    for gt_type, datatype_conf in config["data_types"].items():
        search_input_template, search_output_template = decide_dataset_job_templates(
            input_folder,
            output_folder,
            datatype_conf,
            steps,
            mode="unpack",
        )

        jobs.extend(
            find_jobs(
                input_template=search_input_template,
                output_template=search_output_template,
                gt_type=gt_type,
                subset=subset,
                extra_job_args={
                    "tmp_folder": tmp_folder,
                    "config": datatype_conf,
                    "cpus_per_worker": cpus_per_worker,
                    "loglevel": loglevel,
                },
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

    if args.n_workers is not None and not args.action.endswith("_dataset"):
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
            "copy",
        ],
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        nargs="*",
        choices=["quantize", "pack_video", "unpack_video", "unquantize"],
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
        "--cpus_per_worker",
        type=int,
        default=None,
        help="Number of CPUs per worker for slurm and ffmpeg threads.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        nargs="*",
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
    parser.add_argument(
        "--no-verify-version",
        action="store_true",
        help="Skip verifying our cvdpack version against the version from any input configs",
    )

    return validate_args(parser.parse_args())


def filter_files_by_subset_dict(
    input_files: list[tuple[dict, Path]], subset: dict | None
):
    if subset is None:
        return input_files, 0

    def allowed(file_info: dict):
        return all(
            (
                k not in file_info
                or file_info[k] == v
                or (isinstance(v, (list, set)) and file_info[k] in v)
            )
            for k, v in subset.items()
        )

    res = [
        (file_info, file_path)
        for file_info, file_path in input_files
        if allowed(file_info)
    ]
    skipped = len(input_files) - len(res)

    if len(res) == 0 and len(input_files) > 0:
        logger.warning(
            f"Filtering on {subset=} caused ALL {len(input_files)} files to be skipped"
        )

    return res, skipped


def copy_files(
    input_template: Path,
    output_template: Path,
    subset: dict,
    loglevel: int,
):
    # allow the user to specify no template for EITHER inp or out,
    # in which case we just assume the templates are the same, e.g. for doing subsetting
    match input_template.name == "{}", output_template.name == "{}":
        case False, True:
            input_rel = Path(*input_template.parts[len(output_template.parts) :])
            logger.info(
                f"{output_template=} was a folder, inferring template {output_template / input_rel} based on input"
            )
            output_template = output_template / input_rel
        case True, False:
            output_rel = Path(*output_template.parts[len(input_template.parts) :])
            logger.info(
                f"{input_template=} was a folder, inferring template {input_template / output_rel} based on output"
            )
            input_template = input_template / output_rel
        case False, False:
            pass
        case _:
            raise ValueError(
                f"Invalid {input_template=} {output_template=}, cannot infer the dataset structure"
            )

    input_files = list(match_template_paths(input_template))
    input_files, skipped_for_subset = filter_files_by_subset_dict(input_files, subset)

    if len(input_files) == 0:
        raise ValueError(
            f"No files found matching template: {input_template=} for {subset=}"
        )

    output_files = [
        format_template(output_template, file_info) for file_info, _ in input_files
    ]
    input_files = [f for _, f in input_files]

    uniq_inp = set(input_files)
    uniq_out = set(output_files)
    if len(uniq_inp) != len(uniq_out):
        raise ValueError(
            f"copy from {input_template=} to {output_template=} is not one-to-one, got {len(uniq_inp)=} {len(uniq_out)=}"
        )

    items = zip(input_files, output_files)
    if loglevel <= logging.INFO:
        items = tqdm(items, total=len(input_files))

    for input_file_path, output_file_path in items:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Copying {input_file_path} -> {output_file_path}")
        shutil.copy(input_file_path, output_file_path)


def format_for_json(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main():
    start_time = time.time()

    args = parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(args.loglevel)

    config_path = args.config
    if config_path is None and args.action.endswith("_dataset"):
        config_path = args.input / "cvdpack.json"

    if config_path is not None:
        with config_path.open("r") as f:
            config = json.load(f)
        config_version = config.get("metadata", {}).get("cvdpack_version")
        if config_version is not None and not args.no_verify_version:
            raise ValueError(
                f"Config {config_path} was made for cvdpack version {config_version} which does not match installed cvdpack={__version__} "
                "Please install that version of cvdpack, or use --no-verify-version if you have verified it is safe to skip this check"
            )

    out_suffix = args.output.suffix if not args.output.is_dir() else None
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
                raise ValueError(
                    f"pack_dataset requires input to be a directory: {args.input=}"
                )
            pack_dataset(
                args.input,
                args.output,
                args.steps,
                config,
                args.parallel_mode,
                args.slurm_args,
                args.n_workers,
                args.tmp_folder,
                subset=_parse_k_equals_v_strs(args.subset),
                lazy=args.lazy,
                cpus_per_worker=args.cpus_per_worker,
                loglevel=args.loglevel,
            )
        case "unpack_dataset", _:
            if not args.input.is_dir():
                raise ValueError(
                    f"unpack_dataset requires input to be a directory: {args.input=}"
                )
            unpack_dataset(
                args.input,
                args.output,
                args.steps,
                config,
                args.parallel_mode,
                args.slurm_args,
                args.n_workers,
                subset=_parse_k_equals_v_strs(args.subset),
                tmp_folder=args.tmp_folder,
                lazy=args.lazy,
                cpus_per_worker=args.cpus_per_worker,
                loglevel=args.loglevel,
            )
        case "copy", _:
            copy_files(
                args.input,
                args.output,
                subset=_parse_k_equals_v_strs(args.subset),
                loglevel=args.loglevel,
            )
        case _:
            raise ValueError(f"Invalid {args.action=}")

    if args.action == "copy" or config is None:
        return

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
    elif config_path is not None and not str(config_path).startswith("presets/"):
        with config_path.open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)

    print(
        f"Completed {args.action} for result {args.output} in {time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()
