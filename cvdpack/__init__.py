__version__ = "0.5.3"

compatibility_version = (
    1  # increment for every breaking change which affects the packed/unpacked data
)

# ruff: noqa: E402

from .main import Job, execute_jobs, find_jobs, process_video_job
from .pack_frames import (
    CheckBoundsPacker,
    F16ToInt16ReinterpretPacker,
    F32As2Int16ReinterpretPacker,
    InvQuantizeInt16Packer,
    LinearQuantizeIntPacker,
    UnitSphereAs2F32AnglesPacker,
    UnitSphereAs2Int16Packer,
    get_all_channel_packers,
    get_channel_packer,
)
from .pack_timeseries import pack_video, unpack_video
from .util import (
    format_template,
    match_template_paths,
)

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
    "UnitSphereAs2F32AnglesPacker",
    "UnitSphereAs2Int16Packer",
    "pack_video",
    "unpack_video",
    "Job",
    "find_jobs",
    "process_video_job",
    "execute_jobs",
]
