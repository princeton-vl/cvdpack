__version__ = "0.0.2"

compatibility_version = 1 # increment for every breaking change which affects the packed/unpacked data

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
    OneChannelF32As2Int16ReinterpretPacker,
    F32ToF16ToInt16ReinterpretPacker,
)

from .pack_timeseries import (
    pack_video,
    unpack_video
)
from .main import (
    Job,
    find_jobs,
    process_video_job,
    execute_jobs
)