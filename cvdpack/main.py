#!/usr/bin/env python3

# Copyright (c) 2025, Princeton University
# This code is licensed under the BSD-3-Clause license provided in the root directory of this project.
#
# Authors:
# - Alexander Raistrick <araistrick@princeton.edu>

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import time
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from tqdm import tqdm

from cvdpack import __version__, compatibility_version
from cvdpack import util
from cvdpack import pack_frames
from cvdpack.pack_timeseries import (
    pack_video,
    unpack_video,
    pack_tarball,
    unpack_tarball,
)

try:
    import submitit
except ImportError:
    submitit = None

logger = logging.getLogger("cvdpack")

SLURM_ARRAY_MAX = int(os.environ.get(util.ENVIRON_KEYS["array_max"], 500))


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


@dataclass
class Job:
    input_path: Path
    output_path: Path
    gt_type: str
    subset: dict
    tmp_folder: Path
    config: dict
    cpus_per_worker: int
    loglevel: int


def find_jobs(
    input_template: Path,
    output_template: Path,
    gt_type: str,
    subset: dict | None,
    job_defaults: dict,
    match_video_folder: bool = False,
    lazy: bool = False,
) -> list[Job]:
    if subset is None:
        subset = {}
    elif "gt_type" in subset and subset["gt_type"] != gt_type:
        return []

    logger.debug(
        f"{find_jobs.__name__} {input_template=} {output_template=} {gt_type=} {subset=} {job_defaults=}"
    )

    subset = {**subset, "gt_type": gt_type}
    input_template, matched_keys = util.format_template(
        input_template, subset, return_matched=True
    )
    matched_keys.add("gt_type")

    if match_video_folder and "{frame" in input_template.parts[-1]:
        search_template = input_template.parent
        input_template_extra = input_template.parts[-1]
    else:
        search_template = input_template
        input_template_extra = None

    paths = list(util.match_template_paths(search_template))

    skipped_for_lazy = 0
    jobs = []
    for vid_info, vid_input_path in paths:
        if subset and not util.included_in_filter(
            vid_info, subset, allow_extra=matched_keys
        ):
            continue

        vid_info.update(subset)

        output_path = util.format_template(output_template, vid_info, allow_missing=[])
        if lazy and output_path.exists():
            skipped_for_lazy += 1
            continue

        if input_template_extra:
            extra = util.format_template(input_template_extra, vid_info)
            vid_input_path = vid_input_path / extra

        job = Job(
            input_path=vid_input_path,
            output_path=output_path,
            **job_defaults,
        )
        jobs.append(job)

    if len(jobs) == 0 and skipped_for_lazy == 0:
        raise ValueError(f"No jobs found for {input_template}")
    msg = f"Found {len(jobs)} jobs for {input_template} -> {output_template}"
    if skipped_for_lazy > 0:
        msg += f", skipped {skipped_for_lazy} due to --lazy flag"
    logger.info(msg)

    return jobs


def _process_video(
    input_path: Path,
    output_path: Path,
    job: Job,
    tmp_folder: Path,
):
    packer = (
        pack_frames.get_channel_packer(job.config.get("packing"))
        if "packing" in job.config
        else None
    )

    frame_start = job.config.get("frame_start", 0)
    frame_step = job.config.get("frame_step", 1)

    match input_path.suffix, output_path.suffix:
        case ".png", ".mkv":
            pack_video(
                input_path,
                output_path,
                frame_start=frame_start,
                frame_step=frame_step,
                n_cpus=job.cpus_per_worker,
                loglevel=job.loglevel,
                tmp_folder=tmp_folder,
            )
        case _, ".mkv":
            tmp_template = tmp_folder / "{frame:06d}.png"
            pack_frames.pack_frameset(
                input_path,
                tmp_template,
                packer=packer,
            )
            pack_video(
                tmp_template,
                output_path,
                frame_start=frame_start,
                frame_step=frame_step,
                n_cpus=job.cpus_per_worker,
                loglevel=job.loglevel,
                tmp_folder=tmp_folder,
            )
        case _, ".tar.gz":
            pack_tarball(input_path, output_path)
        case ".mkv", ".png":
            unpack_video(
                input_path,
                output_path,
                n_cpus=job.cpus_per_worker,
                loglevel=job.loglevel,
                tmp_folder=tmp_folder,
            )
        case ".png" | ".jpg" | ".jpeg" | ".npy", ".png":
            pack_frames.pack_frameset(
                input_path,
                output_path,
                packer=packer,
            )
        case ".png", _:
            pack_frames.unpack_frameset(
                input_path,
                output_path,
                packer=packer,
                unpack_channels_last=job.config.get("unpack_channels_last", None),
            )
        case ".mkv", _:
            tmp_frames = tmp_folder / "{frame:06d}.png"
            unpack_video(
                input_path,
                tmp_frames,
                frame_start=frame_start,
                frame_step=frame_step,
                n_cpus=job.cpus_per_worker,
                loglevel=job.loglevel,
                tmp_folder=tmp_folder,
            )
            pack_frames.unpack_frameset(
                tmp_frames,
                output_path,
                packer=packer,
                unpack_channels_last=job.config.get("unpack_channels_last", None),
            )
        case ".tar.gz", _:
            unpack_tarball(input_path, output_path)
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


def process_video_job(job: Job):
    input_path = job.input_path
    output_path = job.output_path

    tmp_root = job.tmp_folder
    out_str = str(output_path)
    out_str = (
        out_str.replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace("_", "-")
        .replace("/", "_")
    )

    tmp_path = tmp_root / out_str

    logger.debug(
        f"Making {tmp_path=} for {input_path=} -> {output_path=}, {tmp_path.exists()=}"
    )
    tmp_path.mkdir(parents=True, exist_ok=False)

    try:
        _process_video(
            input_path,
            output_path,
            job,
            tmp_path,
        )
        print(input_path, output_path)
    finally:
        pass
        # if tmp_path is not None:
        #    shutil.rmtree(tmp_path)


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
                j.result()
                msg = f"Job {j.job_id} completed successfully"
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
    parallel_mode: Literal["multiprocess", "slurm", "none"],
    n_workers: int | None,
    slurm_args: dict | None,
    cpus_per_worker: int | None = None,
):
    logger.info(f"Executing {len(jobs)} jobs with {parallel_mode=} {n_workers=}")

    if n_workers is None:
        for job in jobs:
            func(job)
        return

    match parallel_mode:
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
                logger.debug(f"Updating slurm args: {slurm_args=}")
                if isinstance(n := slurm_args.get("slurm_nodelist"), list):
                    slurm_args["slurm_nodelist"] = ",".join(n)
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
            raise ValueError(f"Invalid {parallel_mode=}")


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
    parallel_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: dict | None,
    n_workers: int,
    tmp_folder: Path,
    subset: dict | None,
    lazy: bool,
    cpus_per_worker: int | None = None,
    loglevel: int | None = None,
):
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

        job_defaults = dict(
            gt_type=gt_type,
            subset=subset,
            tmp_folder=tmp_folder,
            config=datatype_conf,
            cpus_per_worker=cpus_per_worker,
            loglevel=loglevel,
        )

        jobs.extend(
            find_jobs(
                input_template=input_template,
                output_template=output_template,
                gt_type=gt_type,
                subset=subset,
                job_defaults=job_defaults,
                lazy=lazy,
                match_video_folder=True,
            )
        )

    execute_jobs(
        log_folder=output_folder / "logs",
        func=process_video_job,
        jobs=jobs,
        parallel_mode=parallel_mode,
        n_workers=n_workers,
        slurm_args=slurm_args,
        cpus_per_worker=cpus_per_worker,
    )


def unpack_dataset(
    input_folder: Path,
    output_folder: Path,
    steps: list[str] | None,
    config: dict | None,
    parallel_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: dict | None,
    n_workers: int,
    subset: dict | None,
    tmp_folder: Path,
    lazy: bool,
    cpus_per_worker: int | None = None,
    loglevel: int | None = None,
):
    if config is None:
        raise ValueError(
            "unpack_dataset requires a config, must use --config "
            "or use an --input containing a cvdpack.json"
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

        job_defaults = dict(
            gt_type=gt_type,
            subset=subset,
            tmp_folder=tmp_folder,
            config=datatype_conf,
            cpus_per_worker=cpus_per_worker,
            loglevel=loglevel,
        )

        jobs.extend(
            find_jobs(
                input_template=search_input_template,
                output_template=search_output_template,
                gt_type=gt_type,
                subset=subset,
                job_defaults=job_defaults,
                lazy=lazy,
                match_video_folder=True,
            )
        )

    execute_jobs(
        log_folder=output_folder / "logs",
        func=process_video_job,
        jobs=jobs,
        parallel_mode=parallel_mode,
        n_workers=n_workers,
        slurm_args=slurm_args,
        cpus_per_worker=cpus_per_worker,
    )


def validate_args(args: argparse.Namespace):
    if args.config is not None and args.config.parts[0] == "presets":
        args.config = Path(__file__).parent / args.config

    if args.n_workers is not None and args.action == "copy":
        raise ValueError(
            f"{args.parallel_mode=} {args.n_workers=} doesnt currently work for {args.action=}."
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

    if args.parallel_mode == "slurm" and args.n_workers > SLURM_ARRAY_MAX:
        logger.warning(
            f"Requested {args.n_workers=} but only {SLURM_ARRAY_MAX=} can actually be used."
            f"This is because many clusters often limit arrays to 1000 jobs, but the job array e.g. for tartanair is 3000+. "
            f"Set {util.ENVIRON_KEYS['array_max']} to a larger value if this is appropriate forr your cluster"
        )

    if args.tmp_folder is not None:
        args.tmp_folder = args.tmp_folder / f"tmp_{random.randint(0, 10000)}"
        if args.tmp_folder.exists():
            raise FileExistsError(
                f"Temporary folder {args.tmp_folder=} already exists, please delete it or use a different --tmp_folder"
            )

    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"""
        Cvdpack is a tool to reorganize and save space on your computer vision datasets, such as RGB / Depth / Flow / SurfaceNormal framesets or videos.

        Note: you can customize some features by overriding environment variables:
          {list(util.ENVIRON_KEYS.values())}
        """,
    )
    parser.add_argument(
        "action",
        type=str,
        choices=[
            "pack",
            "unpack",
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

    input_files = [
        (tvals, path)
        for tvals, path in util.match_template_paths(input_template)
        if util.included_in_filter(tvals, subset)
    ]

    if len(input_files) == 0:
        raise ValueError(
            f"No files found matching template: {input_template=} for {subset=}"
        )

    output_files = [
        util.format_template(output_template, file_info) for file_info, _ in input_files
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
    if config_path is None:
        config_path = args.input / "cvdpack.json"

    if args.action == "copy" and not config_path.exists():
        config = None
    else:
        with config_path.open("r") as f:
            config = json.load(f)
        config_version = config.get("metadata", {}).get("cvdpack_version")

    compat_version = config.get("metadata", {}).get("compatibility_version", None)
    if compat_version is not None and compat_version != compatibility_version:
        raise ValueError(
            f"Config {config_path} had compatibility version {compat_version} cvdpack=={config_version}"
            f"which is different from installed {compatibility_version} due to cvdpack=={__version__}"
            "This may mean that the config is not compatible with the installed cvdpack, or that the config is outdated"
            "Please install that version of cvdpack, or use --no-verify-version if you have verified it is safe to skip this check"
        )

    if (
        config_version is not None
        and not args.no_verify_version
        and config_version != __version__
    ):
        logger.warning(
            f"Config {config_path} was made for cvdpack version {config_version} which does not match installed cvdpack={__version__} "
            f"This should be safe since {compatibility_version=} matched correctly, but there is a minute chance the compatibility version could be misconfigured"
        )

    subset = util.parse_dictlist_strings(args.subset)
    dataset_jobprocess_kwargs = dict(
        steps=args.steps,
        config=config,
        parallel_mode=args.parallel_mode,
        slurm_args=util.parse_dictlist_strings(args.slurm_args),
        n_workers=args.n_workers,
        tmp_folder=args.tmp_folder,
        subset=subset,
        lazy=args.lazy,
        cpus_per_worker=args.cpus_per_worker,
        loglevel=args.loglevel,
    )

    match args.action:
        case "pack":
            if not args.input.is_dir():
                raise ValueError(
                    f"pack_dataset requires input to be a directory: {args.input=}"
                )
            pack_dataset(args.input, args.output, **dataset_jobprocess_kwargs)
        case "unpack":
            if not args.input.is_dir():
                raise ValueError(
                    f"unpack_dataset requires input to be a directory: {args.input=}"
                )
            unpack_dataset(args.input, args.output, **dataset_jobprocess_kwargs)
        case "copy":
            copy_files(args.input, args.output, subset=subset, loglevel=args.loglevel)
        case _:
            raise ValueError(f"Invalid {args.action=}")

    if args.action == "copy" or config is None:
        return

    config["metadata"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["metadata"]["cvdpack_version"] = __version__
    config["metadata"]["compatibility_version"] = compatibility_version
    config["metadata"]["args"] = vars(args)
    config["metadata"]["environment"] = {
        k: os.environ.get(v) for k, v in util.ENVIRON_KEYS.items()
    }
    config["metadata"]["pack_runtime"] = time.time() - start_time

    if args.action in {"pack", "unpack"}:
        config["metadata"]["original_folder"] = str(args.input)
        config["metadata"]["packed_folder"] = str(args.output)
        logger.info(f"Adding metadata to {args.output / 'cvdpack.json'}")
        with (args.output / "cvdpack.json").open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)

    logger.info(
        f"Completed {args.action} for {args.subset=} {args.output} in {time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()
