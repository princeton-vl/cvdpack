__version__ = "0.0.4"

compatibility_version = (
    1  # increment for every breaking change which affects the packed/unpacked data
)

# ruff: noqa: E402

from .util import (
    match_template_paths,
    format_template,
)

from .pack_frames import (
    get_all_channel_packers,
    get_channel_packer,
    CheckBoundsPacker,
    LinearQuantizeIntPacker,
    InvQuantizeInt16Packer,
    F32As2Int16ReinterpretPacker,
    F16ToInt16ReinterpretPacker,
)

from .pack_timeseries import pack_video, unpack_video
from .main import Job, find_jobs, process_video_job, execute_jobs

__all__ = [
    "match_template_paths",
    "format_template",
    "get_all_channel_packers",
    "get_channel_packer",
    "CheckBoundsPacker",
    "LinearQuantizeIntPacker",
    "InvQuantizeInt16Packer",
    "F32As2Int16ReinterpretPacker",
    "F16ToInt16ReinterpretPacker",
    "pack_video",
    "unpack_video",
    "Job",
    "find_jobs",
    "process_video_job",
    "execute_jobs",
]
