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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def unnormalize_vals(
    img: np.ndarray,
    min_orig_val: float,
    max_orig_val: float,
    to_dtype: np.dtype,
    quantize_method: QuantizeMethod,
) -> np.ndarray:
    match quantize_method:
        case QuantizeMethod.CHECKBOUNDS:
            return img.astype(to_dtype)
        case QuantizeMethod.LINEAR:
            return img.astype(to_dtype) * (max_orig_val - min_orig_val) + min_orig_val
        case QuantizeMethod.INV:
            min_norm = 1 / max_orig_val
            max_norm = 1 / min_orig_val
            img_inv = (img.astype(to_dtype) - min_norm) / (max_norm - min_norm)
            return 1 / img_inv
        case _:
            raise ValueError(f"Invalid {quantize_method=}")


def quantize_frame(
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

    logger.info(
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

    save_any_image(img_quant, output_img_path)


def _curlyframe_to_ffmpeg_frametemplate(input_path: Path, as_glob: bool = False):
    newname = re.sub(
        r"\{frame:(0\d+)d\}",  # e.g. {frame:06d}
        lambda m: "*" if as_glob else f"%{m.group(1)}d",
        input_path.name,
    )
    if "{" in newname:
        raise ValueError(
            f"Input frames path  must not contain templates besides {{frame}}, got {input_path=}"
        )
    return str(input_path.parent / newname)


def unpack_video(
    input_video_path: Path,
    output_frames_path_template: Path,
    ffmpeg: str = "ffmpeg",
):
    logger.info(
        f"{unpack_video.__name__} {input_video_path=} to {output_frames_path_template=}"
    )

    output_path_ffmpeg = _curlyframe_to_ffmpeg_frametemplate(
        output_frames_path_template
    )

    command = f"{ffmpeg} -y -hide_banner -i {input_video_path} {output_path_ffmpeg}"
    logger.info(
        f"Unpacking {input_video_path=} to {output_frames_path_template=}, {command=}"
    )
    subprocess.check_output(command.split())


def pack_video(
    input_frames_path: Path,
    output_video_path: Path,
    ffmpeg: str = "ffmpeg",
):
    logger.info(f"{pack_video.__name__} {input_frames_path=} to {output_video_path=}")

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
    command = f"{ffmpeg} -y -hide_banner -pattern_type glob -i {input_frames_ffmpeg} {encoder_args} -pix_fmt {pix_fmt} -an {output_video_path}"
    logger.info(f"Packing {input_frames_path=} to {output_video_path=}, {command=}")
    subprocess.check_output(command.split())


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

    iter = tqdm(search_folder.rglob(glob_pattern), desc=f"Finding {template=}")
    for p in sorted(list(iter)):
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
    logger.info(
        f"{pack_frameset.__name__} {input_path_template=} to {output_path_template=}"
    )

    all_files = list(match_template_paths(input_path_template))
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

    img_unquant = unnormalize_vals(
        img,
        min_orig_val,
        max_orig_val,
        to_dtype=to_dtype,
        quantize_method=quantize_method,
    )

    isnan = img == np.iinfo(img.dtype).max
    img_unquant[isnan] = np.nan

    min = img[~isnan].min()
    max = img[~isnan].max()
    logger.info(
        f"Unquantizing {input_img_path=} to {output_img_path=}, {isnan.mean()=:.2f}, {min=:.2f}, {max=:.2f}, {img_unquant.min()=:.2f}, {img_unquant.max()=:.2f}"
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
    all_files = match_template_paths(input_path_template)

    for frame_info, frame_input_path in all_files:
        unquantize_frame(
            frame_input_path,
            output_img_path=format_template(output_path_template, frame_info),
            to_dtype=to_dtype,
            quantize_method=quantize_method,
            min_orig_val=min_orig_val,
            max_orig_val=max_orig_val,
        )


def format_template(template: Path, vals: dict):
    def replace_func(match):
        full_spec = match.group(1)
        key = full_spec.split(":")[0]
        if key in vals:
            return ("{" + full_spec + "}").format(**{key: vals[key]})
        return match.group(0)

    res = re.sub(r"\{([^}]+)\}", replace_func, str(template))

    if isinstance(template, Path):
        return Path(res)
    return res


def find_jobs(
    input_template: Path,
    output_template: Path,
    subset: dict,
    extra_job_args: dict,
    match_video_folder: bool = False,
    lazy: bool = False,
):
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
        if subset and not all(
            k not in vid_info or vid_info[k] == v for k, v in subset.items()
        ):
            continue

        output_path = format_template(output_template, vid_info)
        if lazy and output_path.exists():
            skipped_for_lazy += 1
            continue

        if input_template_extra:
            vid_input_path = vid_input_path / input_template_extra

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
    tmp_folder = job["tmp_folder"]
    tmp_folder.mkdir(parents=True, exist_ok=True)

    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Processing {input_path} -> {tmp_path}{output_path}")

    match input_path.suffix, output_path.suffix:
        case ".png", ".mkv":
            pack_video(input_path, output_path)
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
            pack_video(tmp_template, output_path)
        case ".mkv", ".png":
            unpack_video(input_path, output_path)
        case ".mkv", _:
            tmp_frames = tmp_path / "{frame:06d}.png"
            unpack_video(input_path, tmp_frames)
            unpack_frameset(
                tmp_frames,
                output_path,
                to_dtype=DTYPE_MAP[job["config"]["unpack_dtype"]],
                quantize_method=QuantizeMethod.from_str(
                    job["config"]["quantize_method"]
                ),
                min_orig_val=float(job["config"]["min_orig_val"]),
                max_orig_val=float(job["config"]["max_orig_val"]),
            )
        case ".txt", ".npy":
            data = np.loadtxt(input_path)
            assert "{" not in str(output_path), output_path
            np.save(output_path, data)
        case ".npy", ".txt":
            data = np.load(input_path)
            assert "{" not in str(output_path), output_path
            np.savetxt(output_path, data)
        case _:
            raise ValueError(f"Invalid {input_path.suffix=} {output_path.suffix=}")

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
    n_jobs: int,
    slurm_args: list[str],
    tmp_folder: Path,
):
    logger.info(f"Executing {len(jobs)} jobs with {paralell_mode=} {n_jobs=}")

    match paralell_mode:
        case "multiprocess":
            with multiprocessing.Pool(n_jobs) as pool:
                pool.map(process_video_job, jobs)
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
                slurm_cpus_per_task=4,
                slurm_time=60,
                slurm_array_parallelism=n_jobs,
            )
            if slurm_args:
                slurm_args = _parse_k_equals_v_strs(slurm_args)
                executor.update_parameters(**slurm_args)

            pbar = tqdm(total=len(jobs), desc="Running jobs")
            crashed = []
            for i in range(0, len(jobs), SLURM_ARRAY_MAX):
                launched = executor.map_array(
                    process_video_job, jobs[i : i + SLURM_ARRAY_MAX]
                )
                crashed += wait_jobs(launched, pbar)
            if len(crashed) > 0:
                raise ValueError(
                    f"{len(crashed)} jobs crashed, dataset is likely not safe to use. "
                    f"Please check {log_folder} for ID_log.err and ID_log.out for each ID in {crashed}"
                )
        case _:
            for job in jobs:
                process_video_job(job)


def pack_dataset(
    input_folder: Path,
    output_folder: Path,
    config_path: Path | None,
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: list[str],
    n_jobs: int,
    tmp_folder: Path,
    subset: list[str],
    lazy: bool,
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
        jobs.extend(
            find_jobs(
                input_template=input_folder / datatype_conf["original_path_template"],
                output_template=output_folder / datatype_conf["packed_path_template"],
                subset=subset,
                extra_job_args={"tmp_folder": tmp_folder, "config": datatype_conf},
                lazy=lazy,
                match_video_folder=True,
            )
        )

    for i, dc in enumerate(config["data_types"]):
        quantize_method = dc.get("quantize_method", "NONE")
        if quantize_method in ["NONE", "CHECKBOUNDS"]:
            continue
        dc["quantize_precision"] = _calculate_config_precision(
            method=QuantizeMethod.from_str(quantize_method),
            low=float(dc["min_orig_val"]),
            high=float(dc["max_orig_val"]),
            dtype=dc["pack_dtype"],
        )

    execute_jobs(
        log_folder=output_folder / "logs",
        func=process_video_job,
        jobs=jobs,
        paralell_mode=paralell_mode,
        n_jobs=n_jobs,
        slurm_args=slurm_args,
        tmp_folder=tmp_folder,
    )


def unpack_dataset(
    input_folder: Path,
    output_folder: Path,
    config_path: Path | None,
    paralell_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: list[str],
    n_jobs: int,
    subset: list[str],
    tmp_folder: Path,
    lazy: bool,
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
        jobs.extend(
            find_jobs(
                input_template=input_folder / datatype_conf["packed_path_template"],
                output_template=output_folder / datatype_conf["original_path_template"],
                subset=subset,
                extra_job_args={"tmp_folder": tmp_folder, "config": datatype_conf},
                lazy=lazy,
                match_video_folder=True,
            )
        )

    execute_jobs(
        log_folder=output_folder / "logs",
        func=process_video_job,
        jobs=jobs,
        paralell_mode=paralell_mode,
        n_jobs=n_jobs,
        slurm_args=slurm_args,
        tmp_folder=tmp_folder,
    )


def validate_args(args: argparse.Namespace):
    if args.config is not None and args.config.parts[0] == "presets":
        args.config = Path(__file__).parent / args.config

    if args.level == "dataset" and args.tmp_folder is None:
        raise ValueError("Must provide --tmp_folder for dataset packing")

    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        type=str,
        choices=[
            "pack",
            "unpack",
            "check",
            "reorganize",
            "quantize",
            "unquantize",
        ],
    )
    parser.add_argument("level", type=str, choices=["dataset", "scene", "frameset"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

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
        default=None,
        choices=["multiprocess", "slurm", "none"],
        help="Parallelize using a slurm array. Requires cvdpack[slurm] optional dependencies for slurm mode",
    )
    parser.add_argument(
        "--slurm_args",
        type=str,
        default=None,
        nargs="*",
        help="Must be space separated and use key=value format. Passed to submitit executor.update_parameters",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=None, help="Number of jobs to run in parallel."
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
    parser.add_argument("--log_level", type=str, default="INFO")

    return validate_args(parser.parse_args())


def format_for_json(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main():
    args = parse_args()

    start = time.time()

    match args.action, args.level, args.input.suffix, args.output.suffix:
        case "pack", "frameset", _, ".png":
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
        case "pack", "frameset", ".png", ".mkv":
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            pack_video(
                args.input,
                args.output,
            )
        case "unpack", "frameset", ".png", _:
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            unpack_frameset(
                args.input,
                args.output,
                args.to_dtype,
                args.quantize_method,
                args.min_orig_val,
                args.max_orig_val,
            )
        case "unpack", "frameset", ".mkv", ".png":
            args.output.parent.mkdir(parents=True, exist_ok=args.overwrite)
            unpack_video(
                args.input,
                args.output,
            )
        case "pack", "dataset", _, _:
            if not args.input.is_dir():
                raise ValueError(f"Input must be a directory: {args.input=}")

            args.output.mkdir(parents=True, exist_ok=args.overwrite)
            pack_dataset(
                args.input,
                args.output,
                args.config,
                args.parallel_mode,
                args.slurm_args,
                args.n_jobs,
                args.tmp_folder,
                args.subset,
                args.lazy,
            )
        case "unpack", "dataset", _, _:
            if not args.input.is_dir():
                raise ValueError(f"Input must be a directory: {args.input=}")
            args.output.mkdir(parents=True, exist_ok=args.overwrite)
            unpack_dataset(
                args.input,
                args.output,
                args.config,
                args.parallel_mode,
                args.slurm_args,
                args.n_jobs,
                args.subset,
                args.tmp_folder,
                args.lazy,
            )
        case "reorganize", "dataset", _, _:
            targets = match_template_paths(args.input)
        case _:
            raise ValueError(
                f"Invalid {args.action=} {args.level=} {args.input.suffix=} {args.output.suffix=}"
            )

    if args.config is None:
        return

    with args.config.open("r") as f:
        config = json.load(f)

    if args.level == "frameset":
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
    elif args.level == "dataset":
        config["metadata"]["original_folder"] = str(args.input)
        config["metadata"]["packed_folder"] = str(args.output)

    config["metadata"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["metadata"]["cvdpack_version"] = __version__
    config["metadata"]["args"] = vars(args)
    config["metadata"]["pack_runtime"] = time.time() - start

    if args.level == "dataset":
        with (args.output / "cvdpack.json").open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)
    elif args.config is not None and not str(args.config).startswith("presets/"):
        with args.config.open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)


if __name__ == "__main__":
    main()
