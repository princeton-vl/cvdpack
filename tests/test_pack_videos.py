import numpy as np
import pytest
import logging

from cvdpack.pack_timeseries import pack_video, unpack_video
from cvdpack.util import save_any_image, load_any_image
import cvdpack.util as util

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3])
def test_video_pack_unpack_roundtrip(tmp_path, dtype, channels):
    np.random.seed(42)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    unpacked_dir = tmp_path / "unpacked"
    unpacked_dir.mkdir()

    n_frames = 5
    max_val = np.iinfo(dtype).max
    shape = (40, 32) if channels == 1 else (40, 40, channels)
    original_frames = []
    for i in range(n_frames):
        data = np.random.randint(0, max_val + 1, shape, dtype=dtype)
        original_frames.append(data.copy())
        frame_path = frames_dir / f"frame_{i:04d}.png"
        save_any_image(data, frame_path)

    video_path = tmp_path / "test_video.mkv"
    pack_video(
        input_frames_path=frames_dir / "frame_{frame:04d}.png",
        output_video_path=video_path,
        tmp_folder=tmp_path / "tmp",
        loglevel=logging.DEBUG,
    )
    assert video_path.exists(), f"Video file {video_path} was not created"

    out_template = unpacked_dir / "frame_{frame:04d}.png"
    unpack_video(
        input_video_path=video_path,
        output_frames_path_template=out_template,
        tmp_folder=tmp_path / "tmp",
        loglevel=logging.DEBUG,
    )

    out_paths = list(util.match_template_paths(out_template))
    if len(out_paths) != len(original_frames):
        raise ValueError(
            f"Expected {len(original_frames)=} but got {len(out_paths)=} "
            f"got {out_paths=}"
        )

    # Verify equality
    for i in range(len(original_frames)):
        original = original_frames[i]
        unpacked_info, unpacked_path = out_paths[i]
        unpacked = load_any_image(unpacked_path)
        np.testing.assert_array_equal(unpacked, original)
