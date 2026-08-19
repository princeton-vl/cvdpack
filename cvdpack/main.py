#!/usr/bin/env python3

# Copyright (c) 2025, Princeton University
# This code is licensed under the BSD-3-Clause license provided in the root directory of this project.
#
# Authors:
# - Alexander Raistrick <araistrick@princeton.edu>

import argparse
import getpass
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from tqdm import tqdm

from cvdpack import __version__, compatibility_version, huggingface, pack_frames, util
from cvdpack.pack_timeseries import (
    pack_tarball,
    pack_video,
    unpack_tarball,
    unpack_video,
)

try:
    import submitit
except ImportError:
    submitit = None

logger = logging.getLogger("cvdpack")

SLURM_ARRAY_MAX = int(os.environ.get(util.ENVIRON_KEYS["array_max"], 500))
FRAME_FIELDS = {"frame", "framenext"}


@dataclass
class Job:
    input_path: Path
    output_path: Path
    gt_type: str
    tmp_folder: Path | None
    config: dict
    cpus_per_worker: int | None
    loglevel: int | None
    remote: dict | None = None


def _values_agree(a: object, b: object) -> bool:
    # numeric equivalence only bridges a numeric-spec parse; plain text must match
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    a, b = str(a), str(b)
    if a.isdigit() and b.isdigit():
        return int(a) == int(b)
    return a == b


def _discover_groups(
    folder: Path, filename_template: Path, parent_info: dict, group_fields: set[str]
) -> list[dict]:
    groups = {}
    for file_info, _ in util.match_template_paths(folder / filename_template):
        group_info = {k: v for k, v in file_info.items() if k not in FRAME_FIELDS}
        # a repeated field spanning the folder split cannot use the regex backreference
        shared = set(parent_info) & set(group_info)
        if any(not _values_agree(parent_info[k], group_info[k]) for k in shared):
            continue
        info = {**parent_info, **group_info}
        # a numeric-spec parse loses zero padding, so prefer the literal spelling
        spellings = {
            k: parent_info[k] for k in shared if isinstance(parent_info[k], str)
        }
        info.update(spellings)
        key = tuple(info[field] for field in sorted(group_fields))
        groups[key] = info
    return list(groups.values())


def _sequence_paths(
    folder: Path, filename_template: Path, parent_info: dict
) -> list[tuple[dict, Path]]:
    group_fields = util.template_fields(filename_template) - FRAME_FIELDS

    if group_fields <= set(parent_info):
        infos = [parent_info]
    else:
        infos = _discover_groups(folder, filename_template, parent_info, group_fields)

    results = []
    for info in infos:
        filename = util.format_template(
            filename_template, info, allow_missing=list(FRAME_FIELDS)
        )
        results.append((info, folder / filename))
    return results


def find_jobs(
    input_template: Path,
    output_template: Path,
    gt_type: str,
    subset: dict[str, list] | None = None,
    match_video_folder: bool = False,
    lazy: bool = False,
    missing_gt: str = "error",
) -> list[tuple[Path, Path]]:
    if subset is None:
        subset = {}
    subset = {k: util.as_list(v) for k, v in subset.items()}
    if gt_type not in subset.get("gt_type", [gt_type]):
        return []

    logger.debug(
        f"{find_jobs.__name__} {input_template=} {output_template=} {gt_type=} {subset=}"
    )

    subset = {**subset, "gt_type": [gt_type]}
    sequence = match_video_folder and "{frame" in input_template.parts[-1]

    # selecting frames out of a packed sequence is unsupported, so refuse rather than lie
    frame_subset = sorted(k for k in FRAME_FIELDS if k in subset)
    if sequence and frame_subset:
        raise ValueError(
            f"--subset {frame_subset} cannot select frames from {input_template}, "
            "which packs a whole sequence into a single file"
        )

    # a value spanning a path separator cannot match per-component, so it gets its own search
    expandable = sorted(
        k
        for k, v in subset.items()
        if len(v) > 1 and any("/" in str(x) or "\\" in str(x) for x in v)
    )
    if expandable:
        jobs = []
        for value in subset[expandable[0]]:
            narrowed = {**subset, expandable[0]: [value]}
            args = (input_template, output_template, gt_type, narrowed)
            # filters have OR semantics, so only the aggregate decides missing_gt
            jobs += find_jobs(*args, match_video_folder, lazy, "silent")
        if jobs:
            return jobs
        # an empty aggregate is fine under --lazy only if inputs exist and were skipped
        args = (input_template, output_template, gt_type, subset)
        if lazy and find_jobs(*args, match_video_folder, False, "silent"):
            return jobs
        if missing_gt == "error":
            raise ValueError(f"No jobs found for {input_template} with {subset=}")
        if missing_gt == "warn":
            logger.warning(f"No jobs found for {input_template}, skipping")
        return jobs

    # multi-valued keys stay filters; gt_type stays a field so it cannot reopen the greedy match
    fill = {k: v[0] for k, v in subset.items() if len(v) == 1 and k != "gt_type"}
    input_template = util.format_template(input_template, fill)

    if sequence:
        filename_template = Path(input_template.parts[-1])
        folders = [
            (info, path)
            for info, path in util.match_template_paths(input_template.parent)
            if path.is_dir()
        ]
        candidates = []
        for parent_info, parent_path in folders:
            candidates += _sequence_paths(parent_path, filename_template, parent_info)
    else:
        candidates = list(util.match_template_paths(input_template))

    skipped_for_lazy = 0
    jobs = []
    for vid_info, vid_input_path in candidates:
        if not util.included_in_filter(vid_info, subset, allow_extra=set(subset)):
            continue

        vid_info.update(fill)
        vid_info["gt_type"] = gt_type

        output_path = util.format_template(output_template, vid_info)
        if lazy and output_path.exists():
            skipped_for_lazy += 1
            continue

        jobs.append((vid_input_path, output_path))

    if len(jobs) == 0 and skipped_for_lazy == 0:
        if missing_gt == "error":
            raise ValueError(f"No jobs found for {input_template}")
        elif missing_gt == "warn":
            logger.warning(f"No jobs found for {input_template}, skipping")
        return jobs
    msg = f"Found {len(jobs)} jobs for {input_template} -> {output_template}"
    if skipped_for_lazy > 0:
        msg += f", skipped {skipped_for_lazy} due to --lazy flag"
    logger.info(msg)

    return jobs


def ambiguous_job_claims(jobs: list[Job]) -> dict[Path, list[str]]:
    claims: dict[Path, list[str]] = {}
    for job in jobs:
        claims.setdefault(job.input_path, []).append(job.gt_type)
    return {path: names for path, names in claims.items() if len(names) > 1}


def validate_jobs(jobs: list[Job]) -> None:
    outputs = {}
    for job in jobs:
        if job.output_path in outputs:
            raise ValueError(
                f"Jobs {outputs[job.output_path]} and {job.input_path} both write "
                f"to {job.output_path}"
            )
        outputs[job.output_path] = job.input_path


def _unpack_npz_frames(
    input_path: Path,
    output_template: Path,
    frame_start: int,
    frame_step: int,
) -> None:
    data = dict(np.load(input_path))
    shapes = {k: v.shape[0] for k, v in data.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{input_path} has inconsistent timestep dims: {shapes}")
    n_frames = next(iter(shapes.values()))
    for i in range(n_frames):
        frame = frame_start + i * frame_step
        out_path = util.format_template(output_template, {"frame": frame})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame_data = {k: v[i] for k, v in data.items()}
        if out_path.suffix == ".json":
            listified = {k: v.tolist() for k, v in frame_data.items()}
            out_path.write_text(json.dumps(listified))
        else:
            np.savez(out_path.with_suffix(""), **frame_data)


def _process_video(job: Job, tmp_folder: Path) -> None:
    input_path = job.input_path
    output_path = job.output_path

    if "{" not in str(input_path) and not input_path.exists():
        logger.warning(f"Input not found: {input_path}, skipping")
        return

    packer = None
    if "packing" in job.config:
        packer = pack_frames.get_channel_packer(job.config["packing"])

    frame_start = job.config.get("frame_start", 0)
    frame_step = job.config.get("frame_step", 1)

    # each block checks the current representation and advances it toward dst
    cur = input_path
    dst = output_path

    if dst.name.endswith((".tar", ".tar.gz")):
        pack_tarball(cur, dst)
        return
    if cur.name.endswith((".tar", ".tar.gz")):
        unpack_tarball(cur, dst)
        return

    if (
        cur.suffix == ".npz"
        and dst.suffix in (".json", ".npz")
        and "{frame" in str(dst)
    ):
        _unpack_npz_frames(cur, dst, frame_start, frame_step)
        return

    if cur.suffix == ".txt" and dst.suffix == ".npy":
        assert "{" not in str(dst), dst
        np.save(dst, np.loadtxt(cur))
        assert dst.exists(), f"Failed to save {dst=}"
        return
    if cur.suffix == ".npy" and dst.suffix == ".txt":
        assert "{" not in str(dst), dst
        np.savetxt(dst, np.load(cur))
        assert dst.exists(), f"Failed to save {dst=}"
        return

    if cur.suffix == ".mkv" and dst.suffix != ".mkv":
        nxt = dst if dst.suffix == ".png" else tmp_folder / "{frame:06d}.png"
        unpack_video(
            cur,
            nxt,
            frame_start=frame_start,
            frame_step=frame_step,
            n_cpus=job.cpus_per_worker,
            loglevel=job.loglevel,
            tmp_folder=tmp_folder,
        )
        if nxt == dst:
            return
        cur = nxt

    if dst.suffix == ".mkv":
        if cur.suffix != ".png":
            nxt = tmp_folder / "{frame:06d}.png"
            pack_frames.pack_frameset(cur, nxt, packer=packer)
            cur = nxt
        pack_video(
            cur,
            dst,
            frame_start=frame_start,
            frame_step=frame_step,
            n_cpus=job.cpus_per_worker,
            loglevel=job.loglevel,
            tmp_folder=tmp_folder,
        )
        return

    if cur.suffix in (".png", ".jpg", ".jpeg", ".npy") and dst.suffix == ".png":
        pack_frames.pack_frameset(cur, dst, packer=packer)
        return

    if cur.suffix == ".png":
        pack_frames.unpack_frameset(
            cur,
            dst,
            packer=packer,
            unpack_channels_last=job.config.get("unpack_channels_last", None),
        )
        return

    if cur.suffix == dst.suffix:
        if cur.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cur, dst)
        return

    raise ValueError(f"Invalid {input_path.suffix=} {output_path.suffix=}")


def _process_job_in_tmp(job: Job) -> None:
    with tempfile.TemporaryDirectory(dir=job.tmp_folder) as tmp:
        tmp_path = Path(tmp)
        logger.debug(f"Using {tmp_path=} for {job.input_path=} -> {job.output_path=}")
        _process_video(job, tmp_path)


def process_video_job(job: Job) -> None:
    if "{" not in str(job.output_path.parent):
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if job.tmp_folder is not None:
        job.tmp_folder.mkdir(parents=True, exist_ok=True)

    try:
        if job.remote is not None:
            huggingface.fetch_job_input(job.remote, job.input_path)
        _process_job_in_tmp(job)
    finally:
        # a worker-staged download is scratch data, reclaimed even when the job fails
        if job.remote is not None:
            job.input_path.unlink(missing_ok=True)

    print(job.input_path, job.output_path)


def _execute_jobs_slurm(log_folder, func, jobs, n_workers, slurm_args, cpus_per_worker):
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
        crashed += util.wait_jobs(launched, pbar)

    if len(crashed) > 0:
        raise ValueError(
            f"{len(crashed)} jobs crashed, dataset is likely not safe to use. "
            f"Please check {log_folder} for ID_log.err and ID_log.out for each ID in {crashed}"
        )


def execute_jobs(
    log_folder: Path,
    func: Callable,
    jobs: list[dict],
    parallel_mode: Literal["multiprocess", "slurm", "none"],
    n_workers: int | None,
    slurm_args: dict | None,
    cpus_per_worker: int | None = None,
) -> None:
    logger.info(f"Executing {len(jobs)} jobs with {parallel_mode=} {n_workers=}")

    if n_workers is None or parallel_mode == "none":
        for job in jobs:
            func(job)
        return

    if parallel_mode == "multiprocess":
        with multiprocessing.Pool(n_workers) as pool:
            pool.map(func, jobs)
        return
    if parallel_mode == "slurm":
        _execute_jobs_slurm(
            log_folder, func, jobs, n_workers, slurm_args, cpus_per_worker
        )
        return
    raise ValueError(f"Invalid {parallel_mode=}")


STEP_NAMES = {
    "pack": ("pack", "quantize", "pack_video"),
    "unpack": ("unpack", "unpack_video", "unquantize"),
}


def _stage_chain(
    datatype_conf: dict, mode: Literal["pack", "unpack"]
) -> list[tuple[str, Path, Path]]:
    """Stages of (step_name, src_template, dst_template); a run executes a slice."""
    original = Path(datatype_conf["original_path_template"])
    packed = Path(datatype_conf["packed_path_template"])

    if packed.suffix == ".mkv":
        mid = original.with_suffix(".png")
    elif original.suffix == ".txt" and packed.suffix == ".npy":
        mid = original.with_suffix(".npy")
    else:
        mid = None

    if mode == "pack":
        if mid is None:
            return [("pack", original, packed)]
        return [("quantize", original, mid), ("pack_video", mid, packed)]
    if mode == "unpack":
        if mid is None:
            return [("unpack", packed, original)]
        return [("unpack_video", packed, mid), ("unquantize", mid, original)]
    raise ValueError(f"Invalid {mode=}")


def _require_chain_order(
    steps: list[str],
    chain: list[tuple[str, Path, Path]],
    selected: list[tuple[str, Path, Path]],
) -> None:
    names = [name for name, _, _ in chain]
    relevant = [s for s in steps if s in names]
    if relevant != [name for name, _, _ in selected]:
        raise ValueError(
            f"{steps=} lists stages out of order, this data_type runs {names}"
        )


def decide_dataset_job_templates(
    input_folder: Path,
    output_folder: Path,
    datatype_conf: dict,
    steps: list[str] | None,
    mode: Literal["pack", "unpack"],
) -> tuple[Path, Path]:
    """
    By default, run each data_type's whole stage chain, config template to config
    template. A `steps` restriction runs only those stages, so the run starts or
    stops at an intermediate template like quantized pngs.
    """

    assert isinstance(datatype_conf, dict), f"Invalid {datatype_conf=}"
    chain = _stage_chain(datatype_conf, mode)

    if steps is not None and len(steps) == 0:
        raise ValueError("User specified empty --steps, there is no work to be done?")

    unknown = [s for s in steps or [] if s not in STEP_NAMES[mode]]
    if unknown:
        raise ValueError(
            f"Unhandled {unknown=} for {mode=}, valid steps are {STEP_NAMES[mode]}"
        )

    if steps is None or len(chain) == 1:
        selected = chain
    else:
        selected = [stage for stage in chain if stage[0] in steps]
        _require_chain_order(steps, chain, selected)

    if not selected:
        raise ValueError(
            f"Unhandled {steps=} for {mode=}, "
            f"this data_type runs {[name for name, _, _ in chain]}"
        )

    inp = selected[0][1]
    out = selected[-1][2]
    logger.debug(f"{mode=} {steps=} resolved templates {inp=} -> {out=}")
    return input_folder / inp, output_folder / out


def _process_dataset(
    input_folder: Path,
    output_folder: Path,
    mode: Literal["pack", "unpack"],
    steps: list[str] | None,
    config: dict | None,
    parallel_mode: Literal["multiprocess", "slurm", "none"],
    slurm_args: dict | None,
    n_workers: int,
    tmp_folder: Path,
    subset: dict | None,
    lazy: bool,
    missing_gt: str = "error",
    cpus_per_worker: int | None = None,
    loglevel: int | None = None,
    remote: dict | None = None,
) -> None:
    if config is None:
        raise ValueError(
            f"{mode} requires a config, must use --config "
            "or use an --input containing a cvdpack.json"
        )

    jobs = []
    for gt_type, datatype_conf in config["data_types"].items():
        input_template, output_template = decide_dataset_job_templates(
            input_folder, output_folder, datatype_conf, steps, mode
        )
        found = find_jobs(
            input_template=input_template,
            output_template=output_template,
            gt_type=gt_type,
            subset=subset,
            lazy=lazy,
            match_video_folder=True,
            missing_gt=missing_gt,
        )
        args = (gt_type, tmp_folder, datatype_conf, cpus_per_worker, loglevel, remote)
        jobs += [Job(inp, out, *args) for inp, out in found]

    validate_jobs(jobs)
    ambiguous = ambiguous_job_claims(jobs)
    if ambiguous:
        raise ValueError(f"Files matched more than one data_type template: {ambiguous}")
    if not jobs and not lazy:
        raise ValueError(
            f"No data_type template matched anything to {mode} in {input_folder}"
        )

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    execute_jobs(
        log_folder=output_folder / f"{stamp}_cvdpack_{mode}",
        func=process_video_job,
        jobs=jobs,
        parallel_mode=parallel_mode,
        n_workers=n_workers,
        slurm_args=slurm_args,
        cpus_per_worker=cpus_per_worker,
    )


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
    missing_gt: str = "error",
    cpus_per_worker: int | None = None,
    loglevel: int | None = None,
    remote: dict | None = None,
) -> None:
    _process_dataset(
        input_folder,
        output_folder,
        "pack",
        steps=steps,
        config=config,
        parallel_mode=parallel_mode,
        slurm_args=slurm_args,
        n_workers=n_workers,
        tmp_folder=tmp_folder,
        subset=subset,
        lazy=lazy,
        missing_gt=missing_gt,
        cpus_per_worker=cpus_per_worker,
        loglevel=loglevel,
        remote=remote,
    )


# subset/tmp_folder positional order differs from pack_dataset, kept for compatibility
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
    missing_gt: str = "error",
    cpus_per_worker: int | None = None,
    loglevel: int | None = None,
    remote: dict | None = None,
) -> None:
    _process_dataset(
        input_folder,
        output_folder,
        "unpack",
        steps=steps,
        config=config,
        parallel_mode=parallel_mode,
        slurm_args=slurm_args,
        n_workers=n_workers,
        tmp_folder=tmp_folder,
        subset=subset,
        lazy=lazy,
        missing_gt=missing_gt,
        cpus_per_worker=cpus_per_worker,
        loglevel=loglevel,
        remote=remote,
    )


def validate_args(args: argparse.Namespace):
    config_path = args.config
    if config_path is not None and config_path.parts[0] == "presets":
        config_path = Path(__file__).parent / config_path
    hf_source = huggingface.parse_input(args.input)
    input_path = Path(args.input) if hf_source is None else None
    if hf_source is not None and args.action == "pack":
        raise ValueError(
            f"{args.action=} needs local input files but {args.input=} is a huggingface url. "
            "Use unpack to download and unpack, or copy to download the packed files as-is"
        )
    if hf_source is not None and args.action == "unpack" and args.hf_staging is None:
        raise ValueError(
            "A huggingface --input requires choosing --hf_staging: upfront downloads "
            "everything before processing, per_job has each worker download and then "
            "delete only its own inputs"
        )
    upfront_slurm = args.parallel_mode == "slurm" and args.hf_staging == "upfront"
    if hf_source is not None and upfront_slurm and args.tmp_folder is None:
        raise ValueError(
            "--hf_staging upfront with --parallel_mode slurm requires --tmp_folder on "
            "shared storage, since compute nodes cannot read the submit host's default "
            "temporary directory. Omit --hf_staging to let each worker download its "
            "own inputs instead"
        )

    if args.n_workers is not None and args.action == "copy":
        raise ValueError(
            f"{args.parallel_mode=} {args.n_workers=} doesnt currently work for {args.action=}."
        )

    avoids_ffmpeg = args.action == "copy" or (
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

    if args.parallel_mode == "slurm" and (args.n_workers or 0) > SLURM_ARRAY_MAX:
        logger.warning(
            f"Requested {args.n_workers=} but only {SLURM_ARRAY_MAX=} can actually be used."
            f"This is because many clusters often limit arrays to 1000 jobs, but the job array e.g. for tartanair is 3000+. "
            f"Set {util.ENVIRON_KEYS['array_max']} to a larger value if this is appropriate for your cluster"
        )

    tmp_folder = None
    candidates = args.tmp_folder
    if candidates is None and args.action == "unpack":
        # a fixed parent under /tmp would be owned by whichever user ran first
        subfolder = f"cvdpack_{getpass.getuser()}"
        candidates = [Path(tempfile.gettempdir()) / subfolder]
    if candidates is not None:
        tmp_folder = util.select_tmp_folder(candidates, args.min_tmp_folder_space_mb)

    return input_path, config_path, tmp_folder, hf_source


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
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="A local folder, or a huggingface dataset url such as "
        "https://huggingface.co/datasets/<org>/<name>[/tree/<revision>[/<subpath>]]. "
        "Huggingface urls are valid for unpack and copy, and download only what --subset selects.",
    )
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
    parser.add_argument(
        "--tmp_folder",
        type=Path,
        default=None,
        nargs="+",
        help="Scratch folders for intermediate files, the first with enough free space is used. "
        "Unpack defaults to the system temporary directory.",
    )
    parser.add_argument(
        "--hf_staging",
        choices=["upfront", "per_job"],
        default=None,
        help="Whether a huggingface --input is downloaded once before processing, or "
        "fetched by each worker for just its own job and deleted afterwards. "
        "Required when unpacking a huggingface url.",
    )
    parser.add_argument(
        "--min_tmp_folder_space_mb",
        type=int,
        default=0,
        help="Minimum free space in MB required to use a --tmp_folder candidate.",
    )
    parser.add_argument("--lazy", action="store_true", default=False)
    parser.add_argument(
        "--missing_gt",
        choices=["error", "warn", "silent"],
        default="error",
        help="What to do when a gt_type from the config has no matching input files: "
        "'error' (default) raises, 'warn' logs a warning and skips, 'silent' skips quietly",
    )

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

    return parser.parse_args()


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
        if input_file_path.resolve() != output_file_path.resolve():
            shutil.copy(input_file_path, output_file_path)


def format_for_json(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _run_action(
    args: argparse.Namespace,
    input_path: Path,
    subset: dict | None,
    job_kwargs: dict,
    hf_source: tuple | None,
) -> None:
    match args.action:
        case "pack":
            pack_dataset(input_path, args.output, **job_kwargs)
        case "unpack":
            unpack_dataset(input_path, args.output, **job_kwargs)
        case "copy" if hf_source is not None:
            logger.info(f"Downloaded packed files to {args.output}")
        case "copy":
            copy_files(input_path, args.output, subset=subset, loglevel=args.loglevel)
        case _:
            raise ValueError(f"Invalid {args.action=}")


def main():
    start_time = time.time()

    args = parse_args()
    input_path, config_path, tmp_folder, hf_source = validate_args(args)

    logging.basicConfig(
        level=args.loglevel,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(args.loglevel)

    subset = util.parse_dictlist_strings(args.subset)
    hf_remote = None
    hf_staging = None
    if hf_source is not None:
        hf_staging = (
            args.output
            if args.action == "copy"
            else (tmp_folder or args.output) / "hf_download"
        )
    try:
        if hf_source is not None:
            staging = (args, config_path, subset, hf_source, hf_staging)
            input_path, config_path, hf_remote = _stage_remote_input(*staging)
        run_args = (args, input_path, config_path, subset, tmp_folder)
        config = _load_config_and_run(*run_args, hf_source, hf_remote)
    finally:
        # staged downloads are scratch data; the unpacked output stands alone
        if hf_source is not None and args.action != "copy":
            shutil.rmtree(hf_staging, ignore_errors=True)

    if args.action == "copy" or config is None:
        return

    metadata = config.setdefault("metadata", {})
    metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["cvdpack_version"] = __version__
    metadata["compatibility_version"] = compatibility_version
    metadata["args"] = vars(args)
    metadata["environment"] = {
        k: os.environ.get(v) for k, v in util.ENVIRON_KEYS.items()
    }
    metadata["pack_runtime"] = time.time() - start_time

    if args.action in {"pack", "unpack"}:
        # staging folders get deleted, so record the url the data actually came from
        source = args.input if hf_source is not None else str(input_path)
        metadata["original_folder"] = source
        metadata["packed_folder"] = str(args.output)
        logger.info(f"Adding metadata to {args.output / 'cvdpack.json'}")
        with (args.output / "cvdpack.json").open("w") as f:
            json.dump(config, f, indent=2, default=format_for_json)

    logger.info(
        f"Completed {args.action} for {args.subset=} {args.output} in {time.time() - start_time:.2f}s"
    )


def _stage_remote_input(
    args: argparse.Namespace,
    config_path: Path | None,
    subset: dict | None,
    hf_source: tuple,
    hf_staging: Path,
) -> tuple[Path, Path, dict | None]:
    staging_mode = "upfront" if args.action == "copy" else args.hf_staging
    return huggingface.download_input(
        hf_source,
        hf_staging,
        config_path,
        subset,
        compatibility_version,
        staging=staging_mode,
    )


def _load_config_and_run(
    args: argparse.Namespace,
    input_path: Path | None,
    config_path: Path | None,
    subset: dict | None,
    tmp_folder: Path | None,
    hf_source: tuple | None,
    hf_remote: dict | None,
) -> dict | None:
    if config_path is None:
        config_path = input_path / "cvdpack.json"

    local_copy = args.action == "copy" and hf_source is None
    if local_copy and args.config is not None:
        raise ValueError(
            f"--config={args.config} was provided but copy does not read a config"
        )

    # local copy never reads the config; version compatibility matters at unpack time
    if local_copy:
        config = None
        config_version = None
        compat_version = None
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

    # copy_files filters on the --input/--output templates, ignoring any config
    if local_copy:
        subset_keys = util.template_fields(args.input) | util.template_fields(
            args.output
        )
    else:
        subset_keys = util.config_subset_keys(config)
    util.validate_subset_keys(subset, subset_keys)

    # slurm args are scalars for submitit, not multi-valued filters like --subset
    slurm_args = util.parse_dictlist_strings(args.slurm_args)
    if slurm_args is not None:
        slurm_args = {k: v[0] if len(v) == 1 else v for k, v in slurm_args.items()}

    job_kwargs = dict(
        steps=args.steps,
        config=config,
        parallel_mode=args.parallel_mode,
        slurm_args=slurm_args,
        n_workers=args.n_workers,
        tmp_folder=tmp_folder,
        subset=subset,
        lazy=args.lazy,
        missing_gt=args.missing_gt,
        cpus_per_worker=args.cpus_per_worker,
        loglevel=args.loglevel,
        remote=hf_remote,
    )

    if args.action != "copy" and not input_path.is_dir():
        raise ValueError(
            f"{args.action} requires input to be a directory: {input_path=}"
        )

    _run_action(args, input_path, subset, job_kwargs, hf_source)
    return config


if __name__ == "__main__":
    main()
