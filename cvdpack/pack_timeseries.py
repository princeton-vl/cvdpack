import logging
import re
import shutil
import subprocess
import os
import tarfile
from pathlib import Path

from .util import (
    match_template_paths, 
    format_template, 
    ENVIRON_KEYS, 
    load_any_image,
)

logger = logging.getLogger("cvdpack")

ENCODER_ARGS = {
    "ffv1": os.environ.get(
        ENVIRON_KEYS["ffv1_args"],
        "-c:v ffv1 -level 3 -g 1 -slices 4 -threads 4 -slicecrc 1",
    ),
    "libx265": os.environ.get(
        ENVIRON_KEYS["libx265_args"],
        "-c:v libx265 -x265-params lossless=1 -preset slow",
    ),
}

PROPS_TO_ENCODER_PIXFMT = {
    ("uint8", 3): ("libx265", "yuv444p"),
    ("uint16", 1): ("ffv1", "gray16le"),
    ("uint16", 3): ("ffv1", "rgb48"),
    ("uint8", 1): ("libx265", "gray"),
}

FFMPEG = os.environ.get(ENVIRON_KEYS["ffmpeg"], "ffmpeg")
FFMPEG_ARGS = [FFMPEG, "-nostdin", "-y", "-hide_banner"]

def _curlyframe_to_ffmpeg_frametemplate(input_path: Path, as_glob: bool = False):

    """
    Best not to use this function in the general case, because paths like flow force us to use -pattern_type glob, which the ignores -start_number
    """

    if "{frame}" in str(input_path):
        # note RE this warning - we could potentially map {frame} to %04d for ffmpeg, but:
        # (1) it is hard to guess the num digits and
        # (2) this is difficult for cases like TartanAir flow where paths contain both {frame:06d} and {framenext:06d}
        raise ValueError(
            "Direct use of {{frame}} as a template is banned - you must use {{frame:04d}} for some integer. "
            "this is because frame without 0 padding will not be sorted properly when globbed to form a video"
        )

    newname = re.sub(
        r"\{frame:(0\d+)d\}",  # DONT match {framenext:06d} here because we will explictly fix this later
        lambda m: "*" if as_glob else f"%{m.group(1)}d",
        input_path.name,
    )

    logger.debug(
        f"{_curlyframe_to_ffmpeg_frametemplate.__name__} {input_path.name=} -> {newname=}"
    )

    return str(input_path.parent / newname)
def unpack_video(
    input_video_path: Path,
    output_frames_path_template: Path,
    tmp_folder: Path,
    ffmpeg: str = "ffmpeg",
    n_cpus: int | None = None,
    loglevel: int | None = None,
):
    logger.info(
        f"{unpack_video.__name__} {input_video_path=} to {output_frames_path_template=}"
    )
    output_frames_path_template.parent.mkdir(parents=True, exist_ok=True)

    output_path_ffmpeg = _curlyframe_to_ffmpeg_frametemplate(
        output_frames_path_template
    )

    ffmpeg_args = [ffmpeg, "-nostdin","-y", "-hide_banner"]

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

    # to my knowledge, ffmpeg cannot output framenum AND framenum+1 in the output path template,
    # so, we have to go back through and rename all the files to resolve {framenext:...} to frame+1
    if "{framenext" in output_path_ffmpeg:
        targets = list(
            match_template_paths(output_frames_path_template, allow_any=["framenext"])
        )
        for info, file in targets:
            info["framenext"] = info["frame"] + 1
            shutil.move(file, format_template(output_frames_path_template, info))

    return command


def pack_video(
    input_frames_path: Path,
    output_video_path: Path,
    tmp_folder: Path,
    frame_start: int = 0,
    frame_step: int = 1,
    ffmpeg: str = "ffmpeg",
    n_cpus: int | None = None,
    loglevel: int | None = None,
):
        
    logger.info(f"{pack_video.__name__} {input_frames_path=} to {output_video_path=}")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    matched = list(match_template_paths(input_frames_path))
    matched = sorted(matched, key=lambda x: x[0]["frame"])

    first = load_any_image(matched[0][1])
    dim = first.shape[-1] if first.ndim == 3 else 1
    encoder, pix_fmt = PROPS_TO_ENCODER_PIXFMT[(str(first.dtype), dim)]
    encoder_args = ENCODER_ARGS[encoder]

    ffmpeg_args = FFMPEG_ARGS.copy()
    if loglevel != logging.DEBUG:
        ffmpeg_args.extend(["-loglevel", "error"])
        if encoder == "libx265":
            encoder_args = encoder_args + " -x265-params log-level=quiet"
    if n_cpus is not None:
        ffmpeg_args.extend(["-threads", str(n_cpus)])

    # doesnt seem to be recognized for pack?
    #ffmpeg_args.extend(
    #    [
    #        "-start_number",
    #        str(frame_start),
    #    ]
    #)
    
    # These errors are necessary so that we are sure we will unpack to the correct paths on the other end
    # IE we rely wholly on frame_start and frame_step to name the files when unpacking, so they better explain the current filenames correctly.
    for i, (info, path) in enumerate(matched):
        if info["frame"] != frame_start + i * frame_step:
            raise ValueError(
                f"{input_frames_path} had frame {info['frame']} for {i=} {path=}"
                f"but {frame_start=} {frame_step=} means we expected {frame_start + i * frame_step=}"
            )

    input_mode = "txt"
    if input_mode == "txt":
        tmp_folder.mkdir(parents=True, exist_ok=True)
        input_txt_path = tmp_folder / "input.txt"
        assert not input_txt_path.exists(), f"{input_txt_path=} already exists"
        with input_txt_path.open("w") as f:
            for info, path in matched:
                f.write(f"file '{str(path.absolute())}'\n")
        assert input_txt_path.exists(), f"Failed to create {input_txt_path=}"
        ffmpeg_args.extend([
            "-f", "concat", "-safe", "0", "-i", str(input_txt_path.absolute()),
        ])
    elif input_mode == "glob":
        raise NotImplementedError("Globbing input frames can cause incorrect frame numbers since it ignores -start_number")
        ffmpeg_template = _curlyframe_to_ffmpeg_frametemplate(input_frames_path)
        ffmpeg_args.extend([
            "-pattern_type", "glob", ffmpeg_template,
        ])
    else:
        raise ValueError(f"Unknown input_mode {input_mode=}")

    ffmpeg_args.extend(encoder_args.split())
    ffmpeg_args.extend(["-pix_fmt", pix_fmt, "-an", str(output_video_path)])

    command = " ".join(ffmpeg_args)
    logger.info(f"Packing {input_frames_path=} to {output_video_path=}, {command=}")
    subprocess.check_output(ffmpeg_args)

    return command


def pack_tarball(input_frames_template: Path, output_tarball_path: Path):
    logger.info(
        f"{pack_tarball.__name__} {input_frames_template=} to {output_tarball_path=}"
    )
    output_tarball_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_tarball_path, "w:gz") as tar:
        for frame_info, frame_input_path in match_template_paths(input_frames_template):
            output_path = format_template(output_tarball_path, frame_info)
            tar.add(frame_input_path, arcname=output_path.name)


def unpack_tarball(
    input_tarball_path: Path,
    output_frames_path_template: Path,
):
    logger.info(
        f"{unpack_tarball.__name__} {input_tarball_path=} to {output_frames_path_template=}"
    )
    output_frames_path_template.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                tar.extract(member, output_frames_path_template.parent)
